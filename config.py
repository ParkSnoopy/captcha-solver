DEBUG = True

# train.py
USE_GPU = True

DATA_DIR = "./data/ready/"
TRAIN_PERC = 0.90

BATCH_SIZE = 16
NUM_WORKERS = 8
EPOCHS = 20

MODEL_PRETTY_NAME = "CaptchaModel_v3"

# evalu.py
TRAINED_DIR = "./trained/"
RAW_TENSOR = False

# test.py
TEST_DIR = "./data/test/"
#TEST_DIR = "./data/ready/"

# helper.py
# max among `{ (40, 150, 3), (50, 200, 3), (50, 180), (50, 200, 4), (256, 256, 3) }`
MAX_W, MAX_H = (256, 128)
