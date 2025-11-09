from pathlib import Path

import numpy as np
import questionary
import torch
from PIL import Image

from config import DEVICE, RAW_TENSOR, TRAINED_DIR, DATA_DIR, MODEL_CONFIG
from helper import TRANSFORM, fit_image, I2C
from train import USE_MODEL


def _is_file(path: str) -> bool:
    return Path(path).is_file()


def main():
    model_file_path = questionary.path(
        "Select model to use", default=str(TRAINED_DIR), validate=_is_file
    ).ask()

    checkpoint = torch.load(model_file_path, map_location=DEVICE)
    model = USE_MODEL(**MODEL_CONFIG)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(DEVICE).eval()

    while True:
        evaluation_file_path = questionary.path(
            "Select image to evaluate", default=str(DATA_DIR), validate=_is_file
        ).ask()

        img = Image.open(evaluation_file_path)
        img = fit_image(img)
        x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(x)
            if RAW_TENSOR:
                print("\n  < RAW TENSOR >\n", pred, "\n")

        pred = pred.detach().cpu().numpy()
        s = "".join(I2C[int(np.argmax(out))] for out in pred[0])
        print(f"\n  - Pred: `{s}`\n")

        if not questionary.confirm("Continue?", default=True).ask():
            break


if __name__ in {"__main__", "__mp_main__"}:
    main()
