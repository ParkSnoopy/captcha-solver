import numpy as np
from sklearn import (
    preprocessing,
    model_selection,
    metrics,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from pathlib import Path
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo
#from itertools import batched
#import random

from model import (
    CaptchaDataset,
    CaptchaDatasetV3,

    CaptchaModel,
    CaptchaModelV2,
    CaptchaModelV3,
)
from helper import (
#    TRANSFORM,
    do_train,
    do_eval,
    decode_preds,
    dedup,
)
from config import (
    USE_GPU,
    DATA_DIR,

    TRAIN_PERC,
    MAX_W, MAX_H,

    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,

    MODEL_PRETTY_NAME,
)

USE_MODEL   = CaptchaModelV3
USE_DATASET = CaptchaDatasetV3
TIMEZONE    = ZoneInfo("Asia/Beijing")



def main():

    # Device setup
    print("  - Detect device: ", end="")
    if USE_GPU and not torch.cuda.is_available(): raise Exception("No GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Get all images
    print("  - Data Load")
    data_dir = Path(DATA_DIR)

    # matches `.png` only (already converted)
    image_paths = list( data_dir.glob("*.png") )

    print("  - Data PreProcess")
    # `image_paths`->`targets_raw` is `train`->`target` by index
    targets_raw = list( image_path.stem.split('.')[0] for image_path in image_paths )
    targets = [
        [ char for char in target_raw ]
        for target_raw in targets_raw
    ]
    targets_flat = [
        char
        for target in targets
        for char   in target
    ]

    print("  - Fit encoder with target chars")
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(targets_flat)

    targets_encoder = [ label_encoder.transform(target) for target in targets ]
    targets_encoder = np.array( targets_encoder ) + 1

    # Split Data
    print("  - Data Split")
    (
        train_images,
        valid_images,
        train_targets,
        valid_targets,
        _,
        valid_targets_raw,
    ) = model_selection.train_test_split(
        image_paths,
        targets_encoder,
        targets_raw,
        test_size=(1-TRAIN_PERC),
        random_state=(42*42*42*42*42)
    )

    # Ready Dataset and DataLoader
    print("  - Data to Dataset")
    train_dataset = USE_DATASET(
        img_paths=train_images,
        labels   =train_targets,
        img_size =(MAX_W, MAX_H),
    )
    valid_dataset = USE_DATASET(
        img_paths=valid_images,
        labels   =valid_targets,
        img_size =(MAX_W, MAX_H),
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

    # Model, Optim
    print(f"  - Model Ready ( length = `{len(label_encoder.classes_)}` )")
    model = USE_MODEL(
        num_chars = len(label_encoder.classes_),
    )
    model = model.to(device)

    print("  - Optim Ready")
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0005,
        #weight_decay=0.0001,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.8,
        patience=5,
        verbose=True,
    )

    # Train
    print("  - Train Start")
    for epoch in range(EPOCHS):
        print(f"\n    - Train `{epoch+1}`-th epoch")

        train_loss = do_train(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
        )

        print(f"      - Loss   <Training> : `{train_loss:.06f}`")

        valid_preds, valid_loss = do_eval(
            model=model,
            data_loader=valid_loader,
        )

        print(f"      - Loss <Validation> : `{valid_loss:.06f}`")

        valid_preds_char = list()
        for valid_pred in valid_preds:
            current_pred = decode_preds(
                valid_pred,
                label_encoder,
            )
            valid_preds_char.extend(current_pred)

        combined = list(zip(
            valid_targets_raw,
            valid_preds_char,
        ))

        print(combined[:10])

        valid_wo_dup = [ dedup(char) for char in valid_targets_raw ]
        accuracy = metrics.accuracy_score(
            valid_wo_dup,
            valid_preds_char,
        )

        print(f"      - Current Accuracy  : `{accuracy:.06f}`")

        scheduler.step(valid_loss)

        # Save model
        filename = f"./trained/{datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H:%M:%S")}_epoch{epoch+1:02}_on_{MODEL_PRETTY_NAME}.pth"
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
