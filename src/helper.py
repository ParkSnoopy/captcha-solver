import numpy as np
import torchvision.transforms as T
from PIL import Image
from collections import Counter
from typing import Tuple

from config import MAX_W, MAX_H, CHARSET

# Index ↔ Char maps (26 letters + 10 digits)
I2C = {i: k for i, k in enumerate(CHARSET)}
C2I = {k: i for i, k in I2C.items()}

# ImageNet normalization (keep in sync with train/eval)
TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _dominant_corner_color(
    img: Image.Image, _sample: float = 0.05
) -> Tuple[int, int, int]:
    """Why: avoid padding artifacts by matching background."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    c = 1 if arr.ndim == 2 else arr.shape[2]

    pw = max(1, int(w * _sample))
    ph = max(1, int(h * _sample))

    corners = [
        arr[0:ph, 0:pw],  # TL
        arr[0:ph, -pw:],  # TR
        arr[-ph:, 0:pw],  # BL
        arr[-ph:, -pw:],  # BR
    ]
    flat = np.concatenate([corner.reshape(-1, c) for corner in corners], axis=0)
    pixels = [tuple(int(x) for x in rgb) for rgb in flat]
    return Counter(pixels).most_common(1)[0][0] if c == 3 else (0, 0, 0)


def _rgb_from_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def _rgb_from_rgba(img: Image.Image) -> Image.Image:
    bg_rgb = _dominant_corner_color(img.convert("RGB"), _sample=0.1)
    # PIL alpha_composite requires RGBA on both inputs
    bg = Image.new("RGBA", img.size, (*bg_rgb, 255))
    return Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")


def reshape(img: Image.Image) -> Image.Image:
    """Why: center-pad to target canvas, then resize to (MAX_H, MAX_W)."""
    w, h = img.width, img.height
    scale = min(MAX_W / w, MAX_H / h)
    pad_w = int((MAX_W - (w * scale)) // 2)
    pad_h = int((MAX_H - (h * scale)) // 2)
    pad_color = _dominant_corner_color(img, _sample=0.1)

    unify = T.Compose(
        [
            T.Pad((pad_w, pad_h, pad_w, pad_h), fill=pad_color),
            T.Resize((MAX_H, MAX_W)),
        ]
    )
    return unify(img)


def fit_image(img: Image.Image) -> Image.Image:
    """Why: unify all inputs to RGB and fixed canvas before tensor/normalize."""
    if img.mode == "RGBA":
        img = _rgb_from_rgba(img)
    elif img.mode in {"L", "LA"}:
        img = _rgb_from_grayscale(img)
    else:
        img = img.convert("RGB")
    return reshape(img)
