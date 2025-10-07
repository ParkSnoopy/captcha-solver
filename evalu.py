from train import USED_MODEL, USED_WEIGHT, TRANSFORM

import torch
import torch.nn as nn

import questionary
from PIL import Image
from pathlib import Path

USE_GPU = True

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

    model = USED_MODEL(
        weights=USED_WEIGHT.DEFAULT,
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 256*256*1)
    model.load_state_dict(
        torch.load(model_file_path)
    )
    model = model.to(device)

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
        print()
        print(pred)
        print()

    pred_idx = torch.argmax(pred, dim=1)

    print()
    print(pred_idx)
    print()
    print()



if __name__ in {"__main__", "__mp_main__"}:
    main();
