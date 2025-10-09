import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations

from pathlib import Path
from PIL import Image, ImageFile

from helper import (
    _table,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True



class CaptchaDatasetV3:
    def __init__(self, img_paths: list[Path], targets, transform=None):
        # transform = (height, width)
        self.img_paths = img_paths
        self.targets = targets
        self.transform = transform

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
        tgt = self.targets[item]

        if self.transform is not None:
            img = img.resize(
                (self.transform[1], self.transform[0]),
                resample=Image.BILINEAR,
            )

        img = np.array(img)
        aug = self.aug(image=img)
        img = aug["img"]
        img = np.transpose(
            img, (2, 0, 1)
        ).astype(np.float32)

        return {
            "imgs": torch.tensor(img, dtype=torch.float),
            "tgts": torch.tensor(tgt, dtype=torch.long ),
        }

class CaptchaModelV3(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(
            3,
            128,
            kernel_size=(3, 6),
            padding=(1, 1),
        )
        self.pool_1 = nn.MaxPool2d(
            kernel_size=(2, 2),
        )
        self.conv_2 = nn.Conv2d(
            128,
            64,
            kernel_size=(3, 6),
            padding=(1, 1),
        )
        self.pool_2 = nn.MaxPool2d(
            kernel_size=(2, 2),
        )
        self.linear_1 = nn.Linear(
            1152,
            64,
        )
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.GRU(
            64,
            32,
            bidirectional=True,
            num_layers=2,
            dropout=0.25,
            batch_first=True,
        )
        self.output = nn.Linear(
            64,
            num_chars+1,
        )

    def forward(self, imgs, tgts=None):
        bs, _, _, _ = imgs.size()

        x = F      .relu(
               self.conv_1(imgs)
        )
        x = self   .pool_1(x)
        x = F      .relu(
               self.conv_2(x)
        )
        x = self   .pool_2(x)
        x = x      .permute(0, 3, 1, 2)
        x = x      .view(   bs, x.size(1), -1 )
        x = F      .relu(
               self.linear_1(x)
        )
        x = self   .drop_1(x)
        x, _ = self.lstm(x)
        x = self   .output(x)
        x = x      .permute(1, 0, 2)

        if tgts is not None:
            log_probs = F.log_softmax(x, 2)
            inp_len = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            tgt_len = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs, tgts, inp_len, tgt_len
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

        label_idxs = [ _table[c] for c in label ]
        label_tensor = torch.tensor(label_idxs, dtype=torch.long)
        return img, label_tensor

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

#            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
#            nn.ReLU(),
#            nn.MaxPool2d(2),  # Output size: 4x4
#
#            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
#            nn.ReLU(),
#            nn.MaxPool2d(2),  # Output size: 2x2
        )

        # After the CNN, the feature map size is 256 x 16 x 16
        self.cnn_output_size = 512 * 8 * 8
#        self.cnn_output_size = 2048 * 2 * 2

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

class CaptchaModelV2(nn.Module):
    # (26 lowercase letters + 26 uppercase letters + 10 digits = 62 classes)

    def __init__(self, num_classes, captcha_length, input_channels=3, image_width=256, image_height=256):
        super(CaptchaModelV2, self).__init__()

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

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: 4x4

            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: 2x2
        )

        # After the CNN, the feature map size is 256 x 16 x 16
        self.cnn_output_size = 2048 * 2 * 2

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

