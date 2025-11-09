import torchvision.transforms as T
import numpy as np

from PIL import Image
from collections import Counter

from config import (
    MAX_W, MAX_H,
)



I2C = {i:k for i,k in enumerate("QWERTYUIOPASDFGHJKLZXCVBNM1234567890")}
C2I = {k:i for i,k in I2C.items()}

# Transform: fit into pretrained model
TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std =[0.229,0.224,0.225],
    ),
])

def get_dominant_corner_color(img: Image, _sample=0.05):
    img = np.array(img)
    w, h = img.shape[:2]
    channel = 1 if img.ndim == 2 else img.shape[2]

    pw = max(1, int(w * _sample))
    ph = max(1, int(h * _sample))

    corners = list()
    corners.append(img[0:ph, 0:pw]) # TL
    corners.append(img[0:ph, -pw:]) # TB
    corners.append(img[-ph:, 0:pw]) # BL
    corners.append(img[-ph:, -pw:]) # BR

    corners = np.concatenate([corner.reshape(-1, channel) for corner in corners], axis=0)

    pixels = [tuple(rgb) for rgb in corners]
    most_common = Counter(pixels).most_common(1)[0][0]

    return most_common

def rgb_from_grayscale(img) -> Image:
    return img.convert("RGB")
def rgb_from_rgba(img) -> Image:
    bg_color = get_dominant_corner_color(img, _sample=0.1)
    bg = Image.new("RGB", img.size, bg_color)
    return Image.alpha_composite(bg, img).convert('RGB')

def reshape(img: Image) -> Image:
    w = img.width
    h = img.height

    mx = min(
        MAX_W / w,
        MAX_H / h,
    )

    hw = int( ( MAX_W - (w * mx) ) // 2 )
    hh = int( ( MAX_H - (h * mx) ) // 2 )

    unify = T.Compose([
        T.Pad(
            (hw, hh, hw, hh),
            fill=get_dominant_corner_color(img, _sample=0.1),
        ),
        T.Resize(
            (MAX_H,MAX_W)
        ),
    ])

    return unify(img)

def fit_image(img: Image) -> Image:
    # Grayscale
    if len(img.size)==2 or img.size[2]==1:
        img = rgb_from_grayscale(img)
    # RGBA
    elif img.size[2]==4:
        img = rbg_from_rgba(img)

    img = reshape(img)
    return img
