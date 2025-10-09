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
    CaptchaDataset,
    CaptchaModelV21,
)
from helper import (
    TRANSFORM,
)
from config import (
    SEED,
    USE_GPU,
    DATA_DIR,
    TRAIN_PERC,
    EPOCHS,
)

random      .seed(SEED)
np.random   .seed(SEED)
torch.manual_seed(SEED)

MODEL_PRETTY_NAME = "CaptchaModel_v2.1"

USE_MODEL   = CaptchaModelV21
USE_DATASET = CaptchaDataset
TIMEZONE    = ZoneInfo("Asia/Shanghai")



def main():

    # Device setup
    print("  - Detect device: ", end="")
    if USE_GPU and not torch.cuda.is_available(): raise Exception("No GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        lr=0.0004,
    )



    # Train
    print("  - Start Train")
    for epoch in range(EPOCHS):
        print(f"\n    - Training `{epoch+1}` th epoch\n")

        train_loss_value = 0.0

        model.train()
        for images, labels in train_loader:

            images = images.to(device).requires_grad_(False)
            labels = labels.to(device).requires_grad_(False)

            # Optim
            optimizer.zero_grad()
            outputs = model(images)

            # Calc Loss
            train_loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
            )
            train_loss.backward()
            optimizer.step()

            train_loss_value += train_loss.item()

        average_loss = train_loss_value / len(train_loader)

        print(f"      - Loss[Train]: `{average_loss:.06f}`")

        # Validation
        valid_loss_value = 0.0

        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:

                images = images.to(device).requires_grad_(False)
                labels = labels.to(device).requires_grad_(False)

                # Optim
                optimizer.zero_grad()
                outputs = model(images)

                # Calc Loss
                valid_loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1),
                )
                valid_loss.requires_grad = True
                valid_loss.backward()
                optimizer.step()

                valid_loss_value += valid_loss.item()

        valid_loss_value /= len(valid_loader)

        print(f"      - Loss[Eval ]: `{valid_loss_value:.06f}`")

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



if __name__ in {"__main__", "__mp_main__"}:
    main();
