import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image

from helper import (
    _table,
)



# Self-Defined Dataset
class CaptchaDataset(Dataset):
    def __init__(self, img_paths: list[Path], transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = img_path.stem.split('.')[0]

        label_idxs = [ _table[c] for c in label ]
        label_tensor = torch.tensor(label_idxs, dtype=torch.long)
        return img, label_tensor

# CNN + RNN
class CaptchaModel(nn.Module):
    # (26 lowercase letters + 26 uppercase letters + 10 digits = 62 classes)

    def __init__(self, num_classes, captcha_length, input_channels=3, image_width=256, image_height=256):
        super(CaptchaModel, self).__init__()

        # Define CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: 128x128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: 64x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: 32x32

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: 16x16

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: 8x8
        )

        # After the CNN, the feature map size is 256 x 16 x 16
        self.cnn_output_size = 512 * 8 * 8

        # Define 5 separate linear heads, one for each character position
        self.classifier = nn.ModuleList([
            nn.Linear(self.cnn_output_size, num_classes) for _ in range(captcha_length)
        ])

        self.dropout = nn.Dropout(0.5)

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
