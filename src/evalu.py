from pathlib import Path
from random import choices

import numpy as np
import questionary
import torch
from PIL import Image

from config import DEVICE, RAW_TENSOR, TRAINED_DIR, DATA_DIR
from helper import TRANSFORM, fit_image, I2C
from train import USE_MODEL


def _is_file(path: str) -> bool:
    return Path(path).is_file()


def main():
    model_file_path = questionary.path(
        "Select model to use", default=str(TRAINED_DIR), validate=_is_file
    ).ask()

    checkpoint = torch.load(model_file_path, map_location=DEVICE)
    print(checkpoint["args"])
    model = USE_MODEL(**checkpoint["args"]["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(DEVICE).eval()

    evaluation_file_dir = Path(
        questionary.path(
            "Select image directory to evaluate", default=str(DATA_DIR)
        ).ask()
    )
    evaluation_n = int(
        questionary.text(
            "How many images to evaluate",
            default="10",
        ).ask()
    )

    evaluation_file_paths = choices(
        list(evaluation_file_dir.glob("*.png"))
        + list(evaluation_file_dir.glob("*.jpg"))
        + list(evaluation_file_dir.glob("*.jpeg")),
        k=evaluation_n,
    )

    for img_path in evaluation_file_paths:
        img = Image.open(img_path)
        img = fit_image(img)
        x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(x)
            if RAW_TENSOR:
                print("\n  < RAW TENSOR >\n", pred, "\n")

        pred = pred.detach().cpu().numpy()
        s = "".join(I2C[int(np.argmax(out))] for out in pred[0])
        print(f"  - Eval: `{img_path.name}` -> {s}")


if __name__ in {"__main__", "__mp_main__"}:
    main()
