import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
from datetime import datetime
from itertools import batched
import random

USE_GPU = True
DATA_DIR = "./data/ready/"

TRAIN_PERC = 0.90
EPOCHS = 20

USED_MODEL = efficientnet_b1
USED_WEIGHT = EfficientNet_B1_Weights

#OUT_FEATURES = 200000 # 62**5

# Transform: fit into pretrained model (ImageNet)
TRANSFORM = T.Compose([
    #T.Resize(
    #    (224,224)
    #),
    #T.Grayscale( # RGB image is to big to train
    #    num_output_channels=1,
    #),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std =[0.229,0.224,0.225],
    ),
])

#_table = {k:f"{i:02}" for i, k in enumerate("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890")}
_table  = {k:i for i, k in enumerate("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890")}
_rtable = {v:k for k,v in _table.items()}

def _strlabel_to_int(label:str) -> str:
    # DANGER: return `str` as output, not `int`
    return "{}".format(
        ''.join(map(lambda c: _table[c], label))
    );
def _int_to_strlabel(num:int) -> str:
    # int label length is `2*5`
    # because this model is trained for `5-digit` recognition
    num = f"{num}:010"
    return "{}".format(
        ''.join(map(lambda c: _rtable[c[0]+c[1]], batched(num, 2)))
    );

def main():

    # Device setup
    print("  - Detect device: ", end="")
    if USE_GPU and not torch.cuda.is_available(): raise Exception("No GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

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

    # CNN + RNN
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
            )

            # After the CNN, the feature map size is 256 x 16 x 16
            self.cnn_output_size = 512 * 8 * 8

            # Define 5 separate linear heads, one for each character position
            self.classifier = nn.ModuleList([
                nn.Linear(self.cnn_output_size, num_classes) for _ in range(captcha_length)
            ])

            self.dropout = nn.Dropout(0.5)

            '''
            # RNN for sequence prediction
            self.rnn   = nn.LSTM(input_size=1024, hidden_size=256, num_layers=2, batch_first=True)

            # Output layer for 5 characters, each mapped to a class
            self.fc2   = nn.Linear(256, num_chars)
            '''

        def forward(self, x):
            '''
            # CNN layers
            x = self.pool(
                nn.ReLU()(self.conv1(x))
            )
            x = self.pool(
                nn.ReLU()(self.conv2(x))
            )

            # Flatten before feeding into FC layer
            x = x.view(x.size(0), -1)
            x = nn.ReLU()(self.fc1(x))

            # RNN layer for sequence prediction (for 5 characters, we treat it as a sequence)
            x = x.unsqueeze(1)  # Add sequence dimension
            x, _ = self.rnn(x)

            # Decode output to get prediction per character in the sequence
            x = self.fc2(x)

            return x
            '''
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

    # Model, Loss, and Optimizer
    # 26 letters + 26 letters + 10 digits, 256x256 sized, 3 channel (RGB)
    model = CaptchaModel(
        num_classes   =62,
        captcha_length=5,
        input_channels=3,
        image_width   =256,
        image_height  =256,
    )
    model = model.to(device)



    # Get all images
    print("  - Load Dataset")
    data_dir = Path(DATA_DIR)
    img_paths = list(data_dir.glob("*.png"))  # matches `.png` only (already converted)

    # Shuffle for random split
    random.shuffle(img_paths)

    # Split: `TRAIN_PERC` train, `1-TRAIN_PERC` eval
    t_size = int( len(img_paths) * TRAIN_PERC )
    v_size =      len(img_paths) - t_size

    t_imgs = img_paths[:t_size]
    v_imgs = img_paths[t_size:]



    # Ready Dataset and DataLoader
    print("  - Ready Dataset")
    t_dataset = CaptchaDataset(t_imgs, transform=TRANSFORM)
    v_dataset = CaptchaDataset(v_imgs, transform=TRANSFORM)

    t_loader = DataLoader(t_dataset, batch_size=64, shuffle=True)
    v_loader = DataLoader(v_dataset, batch_size=64, shuffle=False)

    '''
    # Define model
    print("  - Define Model, Loss, Optimizer")
    model = USED_MODEL(
        weights=USED_WEIGHT.DEFAULT,
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 256*256*3)
    model = model.to(device)

    # Set Loss and Optim
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )
    '''

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )



    # Train
    print("  - Start Train")
    for epoch in range(EPOCHS):
        print(f"    - Training `{epoch+1}` th epoch")

        model.train()
        running_loss = 0.0

        for images, labels in t_loader:

            images = images.to(device).requires_grad_(False)
            labels = labels.to(device).requires_grad_(False)

            # Optim
            optimizer.zero_grad()
            outputs = model(images)

            # Calc Loss
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(t_loader)

        print(f"      - Training   Loss: `{avg_loss:.09f}`")

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in v_loader:

                images = images.to(device).requires_grad_(False)
                labels = labels.to(device).requires_grad_(False)

                # Optim
                optimizer.zero_grad()
                outputs = model(images)

                # Calc Loss
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1),
                )
                loss.requires_grad = True
                loss.backward()
                optimizer.step()

                val_loss += loss.item()

        val_loss /= len(v_loader)

        print(f"      - Validation Loss: `{val_loss:.09f}`")

        # Save model
        filename = f"./trained/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_epoch{epoch+1:02}_on_self.pth"
        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(
            state,
            filename,
        )
        print(f"      - Model saved as `{filename}`")



if __name__ in {"__main__", "__mp_main__"}:
    main();
