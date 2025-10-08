# train.py
USE_GPU = True
DATA_DIR = "./data/ready/"

TRAIN_PERC = 0.90
EPOCHS = 2

# evalu.py
DEBUG = False
TRAINED_DIR = "./trained/"

# helper.py
# max among `{ (40, 150, 3), (50, 200, 3), (50, 180), (50, 200, 4), (256, 256, 3) }`
MAX_W, MAX_H = (256, 256)
