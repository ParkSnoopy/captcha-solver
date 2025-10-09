import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from pathlib import Path
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo
from itertools import batched
import random

from model import (
    CaptchaDatasetV21,
    CaptchaModelV21,
)
from helper import (
    fit_image,
    TRANSFORM,
    I2C
)
from config import (
    SEED,
    USE_GPU,
    DEVICE,
    DATA_DIR,
    TRAIN_PERC,
    BATCH_SIZE,
    NUM_WORKERS,
    TEST_SIZE,
    EPOCHS,
)

random      .seed(SEED)
np.random   .seed(SEED)
torch.manual_seed(SEED)

MODEL_PRETTY_NAME = "CaptchaModel_v2.1"

USE_MODEL   = CaptchaModelV21
USE_DATASET = CaptchaDatasetV21
TIMEZONE    = ZoneInfo("Asia/Shanghai")



def main():

    # Device setup
    print("  - Detect device: ", end="")
    device = DEVICE
    print(device)

    # Get all images
    print("  - Data Load")
    data_dir = Path(DATA_DIR)
    image_paths = list(data_dir.glob("*.png"))  # matches `.png` only (already converted)

    # Shuffle for random split
    random.shuffle(image_paths)

    # Split: `TRAIN_PERC` train, `1-TRAIN_PERC` evaluation
    train_n = int( len(image_paths) * TRAIN_PERC )
    train_images = image_paths[:train_n]
    valid_images = image_paths[train_n:]

    # Ready Dataset and DataLoader
    print("  - Data to Dataset")
    train_dataset = USE_DATASET(
        img_paths=train_images,
        transform=TRANSFORM,
    )
    valid_dataset = USE_DATASET(
        img_paths=valid_images,
        transform=TRANSFORM,
    )

    print("  - Dataset to DataLoader")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    # Model, Loss, and Optimizer
    # 26 letters 10 digits
    print(f"  - Load Model ( {MODEL_PRETTY_NAME} )")
    model = USE_MODEL(
        num_classes   =36,
        captcha_length=5,
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.0001,
    )

    # Train
    print("  - Train Start")
    for epoch in range(EPOCHS):
        print(f"\n    - Training `{epoch+1}` th epoch")

        train_loss_value = 0.0
        _curr = 0
        _total = len(train_dataset)

        model.train()
        for images, labels in train_loader:

            _step  = len(labels)
            _curr += _step
            print(f"      - Train `{100*_curr/_total:.02f}`% ({_curr}/{_total})", end="\r")

            images = images.to(device).requires_grad_(False)
            labels = labels.to(device).requires_grad_(False)

            optimizer.zero_grad()
            outputs = model(images)

            train_loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
            )
            train_loss.backward()
            optimizer.step()

            train_loss_value += train_loss.item()

        average_loss = train_loss_value / len(train_loader)

        print(f"\n      - Loss[Train]: `{average_loss:.06f}`")

        # Validation
        valid_loss_value = 0.0
        _curr = 0
        _total = len(valid_dataset)

        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:

                _step  = len(labels)
                _curr += _step
                print(f"      - Eval  `{100*_curr/_total:.02f}`% ({_curr}/{_total})", end="\r")

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                valid_loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1),
                )
                #valid_loss.backward()
                #optimizer.step()

                valid_loss_value += valid_loss.item()

        valid_loss_value /= len(valid_loader)

        print(f"\n      - Loss[Valid]: `{valid_loss_value:.06f}`")

        # Test with random samples
        test_images = random.sample(image_paths, k=TEST_SIZE)

        _total = 0;
        _pass  = 0;

        for image_path in test_images:
            _total += 1;

            label = image_path.stem.split('.')[0]
            img = Image.open(image_path)
            img = fit_image(img)
            img = TRANSFORM(img).unsqueeze(0)
            img = img.to(device)
            pred = model(img).detach().cpu().numpy()
            pred = ''.join(map(
                lambda idx: I2C[idx],
                list(map(
                    lambda out: np.argmax(out),
                    pred[0]
                ))
            ))

            if pred.upper() == label.upper():
                _pass += 1;
                print(f"      - Test[{_total:5^}]: PASS ( {pred} ~= {label} ) [ Current Accuracy `{100*_pass/_total:.02f}` % ]", end="\n")
            else:
                print(f"      - Test[{_total:5^}]: FAIL ( {pred} != {label} )", end="\r")

        print(f"\n      -  Accuracy  : `{100*_pass/_total:.02f}` % (Total `{_total}`, Pass `{_pass}`, Fail `{_total-_pass}`)")

        # Save model
        filename = f"./trained/{datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H%M%S")}_epoch{epoch+1:02}_on_{MODEL_PRETTY_NAME}.pth"

        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(
            state,
            filename,
        )

        print(f"\n      - Model saved as `{filename}`")



if __name__ == "__main__": #in {"__main__", "__mp_main__"}:
    main();
