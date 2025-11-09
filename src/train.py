import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

from model import CaptchaDatasetV21, CaptchaModelV21
from helper import fit_image, TRANSFORM, I2C
from config import (
    SEED,
    DEVICE,
    MAX_W,
    MAX_H,
    DATA_DIR,
    TRAIN_PERC,
    BATCH_SIZE,
    NUM_WORKERS,
    TEST_SIZE,
    EPOCHS,
    TRAINED_DIR,
    USE_GPU,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PRETTY_NAME = "CaptchaModel_v2.1"
USE_MODEL = CaptchaModelV21
USE_DATASET = CaptchaDatasetV21
TIMEZONE = ZoneInfo("Asia/Shanghai")


def _ensure_dirs():
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)


def _make_loader(dataset, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=bool(USE_GPU),
        drop_last=False,
    )


def main():
    print("  - Detect device:", DEVICE)
    _ensure_dirs()

    print("  - Data Load")
    data_dir = Path(DATA_DIR)
    image_paths = list(data_dir.glob("*.png"))
    random.shuffle(image_paths)

    train_n = int(len(image_paths) * TRAIN_PERC)
    train_images = image_paths[:train_n]
    valid_images = image_paths[train_n:]

    print("  - Data to Dataset")
    train_dataset = USE_DATASET(
        img_paths=train_images, transform=TRANSFORM, captcha_length=5
    )
    valid_dataset = USE_DATASET(
        img_paths=valid_images, transform=TRANSFORM, captcha_length=5
    )

    print("  - Dataset to DataLoader")
    train_loader = _make_loader(train_dataset, shuffle=True)
    valid_loader = _make_loader(valid_dataset, shuffle=False)

    print(f"  - Load Model ( {MODEL_PRETTY_NAME} )")
    model = USE_MODEL(num_classes=36, captcha_length=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    print("  - Train Start")
    for epoch in range(EPOCHS):
        print(f"\n    - Training `{epoch + 1}` / `{EPOCHS}`")

        model.train()
        running = 0.0
        total_batches = len(train_loader)

        for i, (images, labels) in enumerate(train_loader, 1):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)

            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            running += loss.item()
            if i % 10 == 0 or i == total_batches:
                avg = running / i
                print(f"      - Train {i}/{total_batches} | Loss: {avg:.6f}", end="\r")

        train_loss = running / total_batches
        print(f"\n      - Loss[Train]: `{train_loss:.06f}`")

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                outputs = model(images)
                val_running += criterion(
                    outputs.view(-1, outputs.size(-1)), labels.view(-1)
                ).item()

        valid_loss = val_running / max(1, len(valid_loader))
        print(f"      - Loss[Valid]: `{valid_loss:.06f}`")

        # Probe accuracy on a random subset
        probe_n = min(TEST_SIZE, len(image_paths))
        test_images = random.sample(image_paths, k=probe_n)
        total = 0
        ok = 0
        with torch.no_grad():
            for image_path in test_images:
                total += 1
                label = image_path.stem.split(".")[0]
                img = Image.open(image_path)
                img = fit_image(img)
                x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
                pred = model(x).detach().cpu().numpy()
                pred_str = "".join(I2C[int(np.argmax(out))] for out in pred[0])
                if pred_str.upper() == label.upper():
                    ok += 1
        acc = 100.0 * ok / max(1, total)
        print(
            f"      -  Accuracy  : `{acc:.02f}` % (Total `{total}`, Pass `{ok}`, Fail `{total - ok}`)"
        )

        # Save checkpoint
        filename = (
            TRAINED_DIR
            / f"{datetime.now(tz=TIMEZONE).strftime('%Y%m%d_%H%M%S')}_epoch{epoch + 1:02}_on_{MODEL_PRETTY_NAME}.pth"
        )
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": {"width": MAX_W, "height": MAX_H},
        }
        torch.save(state, str(filename))
        print(f"      - Model saved as `{filename}`")


if __name__ == "__main__":
    main()
