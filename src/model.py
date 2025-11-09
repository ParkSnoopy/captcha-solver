from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

from helper import C2I


class CaptchaDatasetV21(Dataset):
    def __init__(self, img_paths: List[Path], transform=None, captcha_length: int = 5):
        self.img_paths = img_paths
        self.transform = transform
        self.captcha_length = captcha_length

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = img_path.stem.split(".")[0].upper()
        if len(label) != self.captcha_length:
            # Why: mismatched filename-label length destabilizes training.
            raise ValueError(
                f"Label '{label}' length != captcha_length={self.captcha_length} for file {img_path}"
            )
        label_tensor = torch.tensor([C2I[c] for c in label], dtype=torch.long)
        return img, label_tensor


class CaptchaModelV21(nn.Module):
    def __init__(self, num_classes: int = 36, captcha_length: int = 5):
        super().__init__()
        self.captcha_length = captcha_length

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 1024, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(1024, 2048, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # For 250x100 inputs → 2048 × 3 × 7
        cnn_output_size = 43008
        self.classifier = nn.ModuleList(
            [nn.Linear(cnn_output_size, num_classes) for _ in range(captcha_length)]
        )
        self.dropout = nn.Dropout(0.10)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.view(x.size(0), -1)
        feat = self.dropout(feat)
        return torch.stack([head(feat) for head in self.classifier], dim=1)
