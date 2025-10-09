import numpy as np
import torch

import questionary
from PIL import Image
from pathlib import Path

from helper import (
    TRANSFORM,
    fit_image,
    I2C,
)
from config import (
    DEVICE,
    DEBUG,
    RAW_TENSOR,
    TRAINED_DIR,
    DATA_DIR,
    MAX_W, MAX_H,
)
from train import (
    USE_MODEL,
    USE_DATASET,
)



def main():

    device = DEVICE

    def check_is_file(path:str) -> bool:
        return Path(path).is_file();

    model_file_path = questionary.path(
        "Select model to use",
        default=TRAINED_DIR,
        validate=check_is_file,
    ).ask();

    model = USE_MODEL(
        num_classes   =36,
        captcha_length=5,
    )

    try:
        # Latest save format
        model.load_state_dict(
            torch.load(model_file_path)['model']
        )
    except:
        # Legacy save format
        model.load_state_dict(
            torch.load(model_file_path)
        )

    model = model.to(device)

    while True:
        evaluation_file_path = questionary.path(
            "Select image to evaluate",
            default=DATA_DIR,
            validate=check_is_file,
        ).ask();

        # Open, Pad with dominant edge pixel, Resize to (256x128), ToTensor, Normalize
        img = Image.open(evaluation_file_path).convert("RGB")
        img = fit_image(img)
        img = TRANSFORM(img).unsqueeze(0) # `RGB` to `RGBA`
        img = img.to(device)

        with torch.no_grad():
            pred = model(img)

            if RAW_TENSOR:
                print()
                print("  < RAW TENSOR >")
                print(pred)
                print()

        pred = pred.detach().cpu().numpy()
        print()
        print(
            "  - Pred: `{}`".format(
                ''.join(
                    map(
                        lambda idx: I2C[idx],
                        list(map(
                            lambda out: np.argmax(out),
                            pred[0]
                        ))
                    )
                )
            )
        )
        print()

        if not questionary.confirm(
            "Continue?",
            default=True,
        ).ask():
            break;



if __name__ in {"__main__", "__mp_main__"}:
    main();
