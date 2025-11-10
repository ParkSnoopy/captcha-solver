from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

from .helper import C2I
from .config import MAX_W, MAX_H


class CaptchaDatasetV22(Dataset):
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


class ConvBlockV22(nn.Module):
    """Conv→BN→GELU→Dropout. Stride=2 performs downsampling."""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size=3,
        stride=1,
        padding=1,
        dropout=0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(channels_out)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.drop(x)
        return x


class CaptchaModelV22(nn.Module):
    def __init__(
        self,
        n_class: int,
        len_captcha: int,
        blocks=[32, 64, 128, 256],
        dropout=0.1,
    ):
        super().__init__()
        self.len_captcha = len_captcha

        # Start from channel=3 (RGB layer)
        _initial_channel = 3
        self.cnn = nn.Sequential(
            ConvBlockV22(
                channels_in=_initial_channel,
                channels_out=blocks[0],
                dropout=dropout,
                stride=1,
            ),
            *[
                ConvBlockV22(
                    channels_in=blocks[i],
                    channels_out=blocks[i + 1],
                    dropout=dropout,
                    stride=2,
                )
                for i in range(len(blocks) - 1)
            ],
        )
        self.dropout = nn.Dropout(dropout)

        cnn_output_size = self._infer_flatten_dim()
        self.classifier = nn.ModuleList(
            [nn.Linear(cnn_output_size, n_class) for _ in range(len_captcha)]
        )

    @torch.no_grad()
    def _infer_flatten_dim(self) -> int:
        dummy = torch.zeros(1, 3, MAX_H, MAX_W)
        y = self.cnn(dummy)
        return int(y.numel())

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.view(x.size(0), -1)
        feat = self.dropout(feat)
        return torch.stack([head(feat) for head in self.classifier], dim=1)
