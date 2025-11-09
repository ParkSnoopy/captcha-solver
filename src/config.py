from pathlib import Path
from torch import device
from torch.cuda import is_available as have_gpu

DEBUG = True
SEED = 42**3

# Training/eval device selection (respects USE_GPU)
USE_GPU = False
if USE_GPU and not have_gpu():
    raise RuntimeError("USE_GPU=True but no CUDA device is available.")
DEVICE = device("cuda" if (USE_GPU and have_gpu()) else "cpu")

DATA_DIR = Path("./data.smol/")
TRAIN_PERC = 0.90

BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 50

TRAINED_DIR = Path("./checkpoints/")
RAW_TENSOR = False

TEST_SIZE = 100

# Input canvas (W, H)
MAX_W, MAX_H = (250, 100)
