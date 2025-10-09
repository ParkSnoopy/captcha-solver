DEBUG = True
SEED  = 42**3

# train.py
USE_GPU = False

from torch.cuda import is_available as have_gpu
from torch      import device
if USE_GPU and not have_gpu(): raise Exception("No GPU")
DEVICE = device("cuda" if have_gpu() else "cpu")

DATA_DIR = "./data/ready/"
TRAIN_PERC = 0.90
TEST_SIZE = 100

BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 50

# evalu.py
TRAINED_DIR = "./trained/"
RAW_TENSOR = False

# test.py
TEST_DIR = "./data/test/"
#TEST_DIR = "./data/ready/"

# helper.py
# max among `{ (40, 150, 3), (50, 200, 3), (50, 180), (50, 200, 4), (256, 256, 3) }`
MAX_W, MAX_H = (256, 128)
