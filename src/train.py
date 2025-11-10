import random
import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from tqdm.auto import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from model import CaptchaDatasetV22, CaptchaModelV22
from helper import fit_image, TRANSFORM, I2C
from config import (
    SEED,
    MAX_W,
    MAX_H,
    TRAIN_SPLIT,
    BATCH_SIZE,
    WORKERS_N,
    TEST_N,
    DIR_CHECKPOINT,
    CHARSET,
)

USE_MODEL = CaptchaModelV22
USE_DATASET = CaptchaDatasetV22


def _ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _make_loader(cli_args, dataset, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.workers_n,
        pin_memory=cli_args.use_gpu,
        shuffle=shuffle,
        drop_last=False,
    )


def _build_model_config(cli_args):
    return {
        "n_class": len(CHARSET),
        "len_captcha": cli_args.length,
        "blocks": list(map(int, cli_args.blocks.split(","))),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CAPTCHA model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="Directory with training images",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        required=True,
        help="CAPTCHA length",
    )
    parser.add_argument(
        "-bs",
        "--blocks",
        type=str,
        default="32,64,128,512",
        help="CNN blocks (integer separeted by ',')",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=MAX_W,
        help=f"Train image width (default: {MAX_W})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=MAX_H,
        help=f"Train image width (default: {MAX_H})",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0004,
        help="Learning rate",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.05,
        help="Weight decay",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--workers-n",
        type=int,
        default=WORKERS_N,
        help=f"DataLoader worker process count (default: {WORKERS_N})",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=TRAIN_SPLIT,
        help=f"Train split (default: {TRAIN_SPLIT})",
    )
    parser.add_argument(
        "--test-n",
        type=int,
        default=TEST_N,
        help=f"Probe set size after each epoch (default: {TEST_N})",
    )
    parser.add_argument(
        "-o",
        "--save-dir",
        type=Path,
        default=Path(DIR_CHECKPOINT),
        help=f"Directory to write checkpoints (default: {DIR_CHECKPOINT})",
    )
    parser.add_argument(
        "--timezone",
        type=ZoneInfo,
        default=ZoneInfo("Asia/Shanghai"),
        help="Timezone for checkpoint filenames (default: 'Asia/Shanghai')",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="v2.2",
        help="Model name tag for checkpoint filenames (default: 'v2.2')",
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

    return parser.parse_args()


def main(cli_args):
    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)

    device = torch.device("cude" if cli_args.use_gpu else "cpu")

    tqdm.write(f"Detect device: {device}")
    _ensure_dirs(cli_args.save_dir)

    # Data discovery
    tqdm.write("Loading image paths…")
    image_paths = list(cli_args.data.glob("*.png"))
    random.shuffle(image_paths)

    train_n = int(len(image_paths) * cli_args.train_split)
    train_images = image_paths[:train_n]
    valid_images = image_paths[train_n:]

    tqdm.write("Building datasets…")
    train_dataset = USE_DATASET(
        img_paths=train_images, transform=TRANSFORM, captcha_length=cli_args.length
    )
    valid_dataset = USE_DATASET(
        img_paths=valid_images, transform=TRANSFORM, captcha_length=cli_args.length
    )

    tqdm.write("Building dataloaders…")
    train_loader = _make_loader(cli_args=cli_args, dataset=train_dataset, shuffle=True)
    valid_loader = _make_loader(cli_args=cli_args, dataset=valid_dataset, shuffle=False)

    tqdm.write(f"Loading model ({cli_args.name})…")
    model = USE_MODEL(**_build_model_config(cli_args=cli_args)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cli_args.lr,
        weight_decay=cli_args.wd,
    )

    # --- Training loop with tqdm ---
    epoch_bar = trange(cli_args.epochs, desc="Epoch", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{cli_args.epochs}",
            unit="batch",
            leave=False,
        )

        for i, (images, labels) in enumerate(batch_bar, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            running += loss.item()
            avg = running / i
            batch_bar.set_postfix(loss=f"{avg:.6f}")

        train_loss = running / max(1, len(train_loader))
        tqdm.write(f"Loss[Train]: {train_loss:.06f}")

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            val_bar = tqdm(
                valid_loader,
                desc="Valid",
                unit="batch",
                leave=False,
            )
            for images, labels in val_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                batch_loss = criterion(
                    outputs.view(-1, outputs.size(-1)), labels.view(-1)
                ).item()
                val_running += batch_loss
                val_bar.set_postfix(loss=f"{(val_running / max(1, val_bar.n)):.6f}")

        valid_loss = val_running / max(1, len(valid_loader))
        tqdm.write(f"Loss[Valid]: {valid_loss:.06f}")

        # Probe accuracy on a random subset
        probe_n = min(cli_args.test_n, len(image_paths))
        test_images = random.sample(image_paths, k=probe_n)
        total = 0
        ok = 0
        with torch.no_grad():
            probe_bar = tqdm(
                test_images,
                desc="Probe",
                unit="img",
                leave=False,
            )
            for image_path in probe_bar:
                total += 1
                label = image_path.stem.split(".")[0]
                img = Image.open(image_path)
                img = fit_image(img)
                x = TRANSFORM(img).unsqueeze(0).to(device)
                pred = model(x).detach().cpu().numpy()
                pred_str = "".join(I2C[int(np.argmax(out))] for out in pred[0])
                if pred_str.upper() == label.upper():
                    ok += 1
                acc_live = 100.0 * ok / max(1, total)
                probe_bar.set_postfix(acc=f"{acc_live:.2f}%")

        acc = 100.0 * ok / max(1, total)
        tqdm.write(
            f"Accuracy: {acc:.02f}% (Total {total}, Pass {ok}, Fail {total - ok})"
        )

        # Save checkpoint
        filename = (
            cli_args.save_dir
            / f"{datetime.now(tz=cli_args.timezone).strftime('%Y%m%d_%H%M')}_epoch{epoch + 1:02}_on_{cli_args.name}.pth"
        )
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": _build_model_config(cli_args=cli_args),
            "cli_args": cli_args,
        }
        torch.save(state, str(filename))
        tqdm.write(f"Model saved: {filename}")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args=cli_args)
