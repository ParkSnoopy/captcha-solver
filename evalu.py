import numpy as np
import torch

import questionary
from PIL import Image
from pathlib import Path

from model import CaptchaModel
from helper import (
    TRANSFORM,
    _rtable,
    reshape,
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
        num_chars =5,
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

        # Open, Pad with dominant edge pixel, Resize to (256x256), ToTensor, Normalize
        img = Image.open(evaluation_file_path).convert("RGB")
        img = reshape(img)
        img = img.resize(
            (MAX_W, MAX_H),
            resample=Image.BILINEAR,
        )
        img = np.array(img)
        aug = self.aug(image=img)
        img = aug["image"]
        img = np.transpose(
            img, (2, 0, 1)
        ).astype(np.float32)
        img = torch.tensor(img, dtype=torch.float)
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
