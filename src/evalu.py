import random
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import torch
import numpy as np
from PIL import Image

from helper import TRANSFORM, fit_image, I2C
from train import USE_MODEL
from config import SEED


def _is_file(path: Path) -> bool:
    return path.is_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a CAPTCHA model checkpoint on images in a directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help="Path to a `PyTorch` checkpoint",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="Directory of images",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        help="Number of images to evaluate (default: 10)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Use GPU",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose output",
    )
    parser.add_argument(
        "-vv",
        "--raw-tensor",
        action="store_true",
        default=False,
        help="Verbose output (debugging raw tensor)",
    )

    return parser.parse_args()


def main(cli_args):
    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)

    device = torch.device("cuda" if cli_args.use_gpu else "cpu")

    checkpoint = torch.load(cli_args.model, weights_only=False)
    model_conf = checkpoint["model_config"]
    train_args = checkpoint["cli_args"]
    print(model_conf)
    model = USE_MODEL(**model_conf)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    image_paths_as_list = [
        *list(cli_args.data.glob("*.png")),
        *list(cli_args.data.glob("*.jpg")),
        *list(cli_args.data.glob("*.jpeg")),
    ]

    image_paths = random.choices(image_paths_as_list, k=cli_args.num)

    evalu_len = len(image_paths[0].name.split(".")[0])
    train_len = train_args.length
    if evalu_len != train_len:
        raise ValueError(f"Trained for length `{train_len}`, but got `{evalu_len}`")

    progress_bar = tqdm(
        image_paths,
        desc="Evaluate",
        unit="img",
        leave=False,
    )
    for img_path in progress_bar:
        img = Image.open(img_path)
        img = fit_image(img)
        x = TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)
            if cli_args.raw_tensor:
                tqdm.write(f"\n  < RAW TENSOR >\n{pred}\n")

        pred = pred.detach().cpu().numpy()
        s = "".join(I2C[int(np.argmax(out))] for out in pred[0])
        tqdm.write(f"  `{img_path.name}` -> `{s}`")


if __name__ in {"__main__", "__mp_main__"}:
    cli_args = parse_args()
    main(cli_args=cli_args)
