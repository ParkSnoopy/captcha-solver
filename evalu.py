from train import CaptchaModel, TRANSFORM, _rtable

import numpy as np
import torch

import questionary
from PIL import Image
from pathlib import Path

USE_GPU = True
DEBUG = False

TRAINED_MODEL_PATH = "./trained/"
EVALU_IMAGE_PATH = "./data/ready/"



def main():
    if USE_GPU and not torch.cuda.is_available(): raise Exception("No GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_is_file(path:str) -> bool:
        return Path(path).is_file();

    model_file_path = questionary.path(
        "Select model to use",
        default=TRAINED_MODEL_PATH,
        validate=check_is_file,
    ).ask();

    model = CaptchaModel(
        num_classes   =62,
        captcha_length=5,
        input_channels=3,
        image_width   =256,
        image_height  =256,
    )
    model.load_state_dict(
        torch.load(model_file_path)
    )
    model = model.to(device)

    while True:
        evaluation_file_path = questionary.path(
            "Select image to evaluate",
            default=EVALU_IMAGE_PATH,
            validate=check_is_file,
        ).ask();

        img = Image.open(evaluation_file_path)
        img = TRANSFORM(img).unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            pred = model(img)

            if DEBUG:
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
                        lambda idx: _rtable[idx],
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
