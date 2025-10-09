import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image

from helper import (
    C2I,
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
                3, 32,
                kernel_size=(3, 6),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),

            nn.Conv2d(
                32, 64,
                kernel_size=(3, 6),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),

            nn.Conv2d(
                64, 128,
                kernel_size=(3, 6),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),

             nn.Conv2d(
                128, 256,
                kernel_size=(3, 6),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),

            nn.Conv2d(
                256, 512,
                kernel_size=(3, 6),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),

            nn.Conv2d(
                512, 4096,
                kernel_size=(3, 6),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),
        )

        cnn_output_size = 512 * 8 * 8

        # Define 5 separate linear heads, one for each character position
        self.classifier = nn.ModuleList([
            nn.Linear(cnn_output_size, num_classes) for _ in range(captcha_length)
        ])

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        x = self.cnn(x)
        x = x.view(
            x.size(0),
            -1,
        )
        x = self.dropout(x)

        outputs = torch.stack([
            head(x)
            for head in self.classifier
        ], dim=1)

        return outputs
