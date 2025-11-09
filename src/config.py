from pathlib import Path
from torch import device
from torch.cuda import is_available as have_gpu

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

DEBUG = True
SEED = 42**3

# Training/eval device selection (respects USE_GPU)
USE_GPU = False
if USE_GPU and not have_gpu():
    raise RuntimeError("USE_GPU=True but no CUDA device is available.")
DEVICE = device("cuda" if (USE_GPU and have_gpu()) else "cpu")

DATA_DIR = Path("./dist5/")
TRAIN_PERC = 0.90

BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 20

TRAINED_DIR = Path("./checkpoints/")
RAW_TENSOR = False

TEST_SIZE = 100

# Input canvas (W, H)
MAX_W, MAX_H = (250, 100)
LENGTH = 5

MODEL_CONFIG = {
    "n_class": len(CHARSET),
    "len_captcha": LENGTH,
    #          100 50   25   12    6
    "blocks": [64, 128, 256, 1024, 2048],
}
