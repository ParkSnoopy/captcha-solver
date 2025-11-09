import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image, ImageFile

from helper import (
    C2I,
)
from config import (
    MAX_W, MAX_H,
)



# Self-Defined Dataset
class CaptchaDatasetV21(Dataset):
    def __init__(self, img_paths: list[Path], transform=None):
        self.img_paths = img_paths
        self.transform = transform

        self._item_len = len(img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = img_path.stem.split('.')[0].upper()

        label_idxs = [ C2I[c] for c in label ]
        label_tensor = torch.tensor(label_idxs, dtype=torch.long)
        return img, label_tensor

class CaptchaModelV21(nn.Module):
    # 26 letters + 10 digits
    def __init__(self, num_classes=36, captcha_length=5):
        super(CaptchaModelV21, self).__init__()

        # Define CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(
                32,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),

            nn.Conv2d(
                64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),

            nn.Conv2d(
                256,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),

            nn.Conv2d(
                1024,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),

            nn.Conv2d(
                2048,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),
        )

        cnn_output_size = 131072

        # Define 5 separate linear heads, one for each character position
        self.classifier = nn.ModuleList([
            nn.Linear(cnn_output_size, num_classes) for _ in range(captcha_length)
        ])

        self.dropout = nn.Dropout(0.10)

    def forward(self, x):

        features = self.cnn(x)
        features = features.view(
            x.size(0),
            -1,
        )
        features = self.dropout(features)

        outputs = torch.stack([
            head(features)
            for head in self.classifier
        ], dim=1)

        return outputs
