import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from pathlib import Path
from PIL import Image
from datetime import datetime
from itertools import batched
import random

from model import (
    CaptchaDataset,
    CaptchaModel,
)
from helper import (
    TRANSFORM,
)
from config import (
    USE_GPU,
    DATA_DIR,
    TRAIN_PERC,

    EPOCHS,
)



def main():

    # Device setup
    print("  - Detect device: ", end="")
    if USE_GPU and not torch.cuda.is_available(): raise Exception("No GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Model, Loss, and Optimizer
    # 26 letters + 26 letters + 10 digits, 256x256 sized, 3 channel (RGB)
    model = CaptchaModel(
        num_classes   =62,
        captcha_length=5,
        input_channels=3,
        image_width   =256,
        image_height  =256,
    )
    model = model.to(device)



    # Get all images
    print("  - Load Dataset")
    data_dir = Path(DATA_DIR)
    img_paths = list(data_dir.glob("*.png"))  # matches `.png` only (already converted)

    # Shuffle for random split
    random.shuffle(img_paths)

    # Split: `TRAIN_PERC` train, `1-TRAIN_PERC` eval
    t_size = int( len(img_paths) * TRAIN_PERC )
    v_size =      len(img_paths) - t_size

    t_imgs = img_paths[:t_size]
    v_imgs = img_paths[t_size:]



    # Ready Dataset and DataLoader
    print("  - Ready Dataset")
    t_dataset = CaptchaDataset(t_imgs, transform=TRANSFORM)
    v_dataset = CaptchaDataset(v_imgs, transform=TRANSFORM)

    t_loader = DataLoader(t_dataset, batch_size=64, shuffle=True)
    v_loader = DataLoader(v_dataset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )



    # Train
    print("  - Start Train")
    for epoch in range(EPOCHS):
        print(f"    - Training `{epoch+1}` th epoch")

        model.train()
        running_loss = 0.0

        for images, labels in t_loader:

            images = images.to(device).requires_grad_(False)
            labels = labels.to(device).requires_grad_(False)

            # Optim
            optimizer.zero_grad()
            outputs = model(images)

            # Calc Loss
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(t_loader)

        print(f"      - Training   Loss: `{avg_loss:.09f}`")

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in v_loader:

                images = images.to(device).requires_grad_(False)
                labels = labels.to(device).requires_grad_(False)

                # Optim
                optimizer.zero_grad()
                outputs = model(images)

                # Calc Loss
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1),
                )
                loss.requires_grad = True
                loss.backward()
                optimizer.step()

                val_loss += loss.item()

        val_loss /= len(v_loader)

        print(f"      - Validation Loss: `{val_loss:.09f}`")

        # Save model
        filename = f"./trained/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_epoch{epoch+1:02}_on_self.pth"
        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(
            state,
            filename,
        )
        print(f"      - Model saved as `{filename}`")



if __name__ in {"__main__", "__mp_main__"}:
    main();
