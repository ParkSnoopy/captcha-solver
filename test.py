import numpy as np
import torch

import questionary
from PIL import Image
from pathlib import Path

#from model import CaptchaModel, CaptchaModelV2
from train import USE_MODEL
from helper import (
    TRANSFORM,
    _rtable,
    reshape,
)
from config import (
    DEBUG,
    RAW_TENSOR,
    USE_GPU,
    TRAINED_DIR,
    DATA_DIR,
    TEST_DIR,
)



def main():
    if USE_GPU and not torch.cuda.is_available(): raise Exception("No GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_paths = list( Path(TRAINED_DIR).glob("*.pth") )
    model_file_paths.sort()

    for model_file_path in model_file_paths:
        print(f"  - Model: `{model_file_path}`")
        model = USE_MODEL(
            num_classes   =62,
            captcha_length=5,
            input_channels=3,
            image_width   =256,
            image_height  =256,
        )

        try:
            # Legacy save format
            model.load_state_dict(
                torch.load(model_file_path)
            )
        except:
            # Latest save format
            model.load_state_dict(
                torch.load(model_file_path)['model']
            )

        model = model.to(device)

        _total = 0;
        _pass  = 0;

        test_files = list()
        test_files.extend(
            Path(TEST_DIR).glob("*.png")
        )
        test_files.extend(
            Path(TEST_DIR).glob("*.jpg")
        )

        for test_file in test_files:
            _total += 1;

            label = test_file.stem.split('.')[0]
            img = Image.open(test_file)
            img = reshape(img, _indent=2)
            img = TRANSFORM(img).unsqueeze(0)
            img = img.to(device)
            pred = model(img).detach().cpu().numpy()
            pred = ''.join(map(
                lambda idx: _rtable[idx],
                list(map(
                    lambda out: np.argmax(out),
                    pred[0]
                ))
            ))

            if DEBUG:
                if pred.upper() == label.upper():
                    _pass += 1;
                    print(f"    - Test `{_total}`: PASS ( {pred} ~= {label} ) [ Current Accuracy `{100*_pass/_total:.02f}` % ]", end="\n")
                else:
                    print(f"    - Test `{_total}`: FAIL ( {pred} != {label} )", end="\r")

        print(f"\n      - Overall Accuracy: `{100*_pass/_total:.02f}` % (Total `{_total}`, Pass `{_pass}`, Fail `{_total-_pass}`)\n")



if __name__ in {"__main__", "__mp_main__"}:
    main();
