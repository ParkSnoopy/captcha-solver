import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations

from pathlib import Path
from PIL import Image, ImageFile

from helper import (
    C2I,
)
from config import (
    MAX_W, MAX_H,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True



class CaptchaDatasetV3:
    def __init__(self, img_paths: list[Path], labels, img_size=None):
        # img_size = (width, height)
        self.img_paths = img_paths
        self.labels = labels
        self.img_size = img_size

        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                )
            ]
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, item) -> dict:
        img = Image.open(self.img_paths[item]).convert("RGB")
        label = self.labels[item]

        if self.img_size is not None:
            img = img.resize(
                self.img_size,
                resample=Image.BILINEAR,
            )

        img = np.array(img)
        aug = self.aug(image=img)
        img = aug["image"]
        img = np.transpose(
            img, (2, 0, 1)
        ).astype(np.float32)

        return {
            "images" : torch.tensor(img  , dtype=torch.float),
            "targets": torch.tensor(label, dtype=torch.long ),
        }

class CaptchaModelV3(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModelV3, self).__init__()
        self.conv_1 = nn.Conv2d(
            3, 128,
            kernel_size=(3, 6),
            padding=(1, 1),
        )
        self.pool_1 = nn.MaxPool2d(
            kernel_size=(2, 2),
        )
        self.conv_2 = nn.Conv2d(
            128, 64,
            kernel_size=(3, 6),
            padding=(1, 1),
        )
        self.pool_2 = nn.MaxPool2d(
            kernel_size=(2, 2),
        )
        self.linear_1 = nn.Linear(
            2048, 64,
        )
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.GRU(
            64, 32,
            bidirectional=True,
            num_layers=2,
            dropout=0.25,
            batch_first=True,
        )
        self.output = nn.Linear(
            64,
            num_chars+1,
        )

    def forward(self, images, targets=None):
        bs, _, _, _ = images.size()

        x = F      .relu(
            self   .conv_1(images)
        )
        x = self   .pool_1(x)
        x = F      .relu(
            self   .conv_2(x)
        )
        x = self   .pool_2(x)
        x = x      .permute(0, 3, 1, 2)
        x = x      .view(   bs, x.size(1), -1 )
        x = F      .relu(
            self   .linear_1(x)
        )
        x = self   .drop_1(x)
        x, _ = self.lstm(x)
        x = self   .output(x)
        x = x      .permute(1, 0, 2)

        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            input_len = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_len = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_len, target_len
            )
            return x, loss

        return x, None



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

