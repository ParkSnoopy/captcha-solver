import torchvision.transforms as T
import numpy as np

from PIL import Image
from collections import Counter

from config import (
    MAX_W, MAX_H, DEBUG,
)



#_table = {k:f"{i:02}" for i, k in enumerate("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890")}
_table  = {k:i for i, k in enumerate("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890")}
_rtable = {i:k for k,i in _table.items()}

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

    return most_common;

def rgb_from_grayscale(img) -> Image:
    return img.convert("RGB")
def rgb_from_rgba(img) -> Image:
    bg_color = get_dominant_corner_color(img, _sample=0.1)
    bg = Image.new("RGB", img.size, bg_color)
    return Image.alpha_composite(bg, img).convert('RGB')

def reshape(img: Image, _indent=1) -> Image:
    # If image is too small, enlarge it
    if img.width < 130:
        mx = 150 / img.width
        if DEBUG:
            print(f"{'  '*_indent}- Resizing: `{mx:.3f}`x")
        img = T.Resize(
            (
                int(mx * img.width),
                int(mx * img.height),
            )
        )(img)
    # If image is too large, make it smaller
    if img.width > 256:
        mx = 256 / img.width
        if DEBUG:
            print(f"{'  '*_indent}- Resizing: `{mx:.3f}`x")
        img = T.Resize(
            (
                int(mx * img.width),
                int(mx * img.height),
            )
        )(img)

    # Grayscale
    if len(img.size)==2 or img.size[2]==1:
        img = rgb_from_grayscale(img)
    # RGBA
    elif img.size[2]==4:
        img = rbg_from_rgba(img)

    hw = (MAX_W-img.width )//2
    hh = (MAX_H-img.height)//2

    unify = T.Compose([
        T.Pad(
            (hw, hh, hw, hh),
            fill=get_dominant_corner_color(img, _sample=0.1),
        ),
        T.Resize(
            (256,256)
        ),
    ])

    return unify(img);
