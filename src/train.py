import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from tqdm.auto import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from model import CaptchaDatasetV22, CaptchaModelV22
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
    LENGTH,
    MODEL_CONFIG,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PRETTY_NAME = "v2.2"
USE_MODEL = CaptchaModelV22
USE_DATASET = CaptchaDatasetV22
TIMEZONE = ZoneInfo("Asia/Shanghai")


def _ensure_dirs():
    Path(TRAINED_DIR).mkdir(parents=True, exist_ok=True)


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
    tqdm.write(f"Detect device: {DEVICE}")
    _ensure_dirs()

    # Data discovery
    tqdm.write("Loading image paths…")
    data_dir = Path(DATA_DIR)
    image_paths = list(data_dir.glob("*.png"))
    random.shuffle(image_paths)

    train_n = int(len(image_paths) * TRAIN_PERC)
    train_images = image_paths[:train_n]
    valid_images = image_paths[train_n:]

    tqdm.write("Building datasets…")
    train_dataset = USE_DATASET(
        img_paths=train_images, transform=TRANSFORM, captcha_length=LENGTH
    )
    valid_dataset = USE_DATASET(
        img_paths=valid_images, transform=TRANSFORM, captcha_length=LENGTH
    )

    tqdm.write("Building dataloaders…")
    train_loader = _make_loader(train_dataset, shuffle=True)
    valid_loader = _make_loader(valid_dataset, shuffle=False)

    tqdm.write(f"Loading model ({MODEL_PRETTY_NAME})…")
    model = USE_MODEL(**MODEL_CONFIG).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # --- Training loop with tqdm ---
    epoch_bar = trange(EPOCHS, desc="Epoch", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{EPOCHS}",
            unit="batch",
            leave=False,
        )

        for i, (images, labels) in enumerate(batch_bar, 1):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            running += loss.item()
            avg = running / i
            batch_bar.set_postfix(loss=f"{avg:.6f}")

        train_loss = running / max(1, len(train_loader))
        tqdm.write(f"Loss[Train]: {train_loss:.06f}")

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            val_bar = tqdm(
                valid_loader,
                desc="Valid",
                unit="batch",
                leave=False,
            )
            for images, labels in val_bar:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                outputs = model(images)
                batch_loss = criterion(
                    outputs.view(-1, outputs.size(-1)), labels.view(-1)
                ).item()
                val_running += batch_loss
                val_bar.set_postfix(loss=f"{(val_running / max(1, val_bar.n)):.6f}")

        valid_loss = val_running / max(1, len(valid_loader))
        tqdm.write(f"Loss[Valid]: {valid_loss:.06f}")

        # Probe accuracy on a random subset
        probe_n = min(TEST_SIZE, len(image_paths))
        test_images = random.sample(image_paths, k=probe_n)
        total = 0
        ok = 0
        with torch.no_grad():
            probe_bar = tqdm(
                test_images,
                desc="Probe",
                unit="img",
                leave=False,
            )
            for image_path in probe_bar:
                total += 1
                label = image_path.stem.split(".")[0]
                img = Image.open(image_path)
                img = fit_image(img)
                x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
                pred = model(x).detach().cpu().numpy()
                pred_str = "".join(I2C[int(np.argmax(out))] for out in pred[0])
                if pred_str.upper() == label.upper():
                    ok += 1
                acc_live = 100.0 * ok / max(1, total)
                probe_bar.set_postfix(acc=f"{acc_live:.2f}%")

        acc = 100.0 * ok / max(1, total)
        tqdm.write(
            f"Accuracy: {acc:.02f}% (Total {total}, Pass {ok}, Fail {total - ok})"
        )

        # Save checkpoint
        filename = (
            TRAINED_DIR
            + f"{datetime.now(tz=TIMEZONE).strftime('%Y%m%d_%H%M%S')}_epoch{epoch + 1:02}_on_{MODEL_PRETTY_NAME}.pth"
        )
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": {"width": MAX_W, "height": MAX_H, "model_config": MODEL_CONFIG},
        }
        torch.save(state, str(filename))
        tqdm.write(f"Model saved: {filename}")


if __name__ == "__main__":
    main()
