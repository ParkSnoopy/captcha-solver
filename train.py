import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
from datetime import datetime
import random

DATA_DIR = "./data/ready/"
TRAIN_PERC = 0.90
EPOCHS = 2



# Manually merged dataset
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

        # Dummy labels (0 for all) since we have no classes
        label = 0
        return img, label



# Device setup
print("  - Detect device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transform: fit into pretrained model (ImageNet)
transform = T.Compose([
    T.Resize(
        (224,224)
    ),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std =[0.229,0.224,0.225],
    ),
])

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
t_dataset = CaptchaDataset(t_imgs, transform=transform)
v_dataset = CaptchaDataset(v_imgs, transform=transform)

t_loader = DataLoader(t_dataset, batch_size=32, shuffle=True)
v_loader = DataLoader(v_dataset, batch_size=32, shuffle=False)

# Define model
print("  - Define Model, Loss, Optimizer")
model = efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT,
)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(device)

# Set Loss and Optim
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



# Train
print("  - Train")
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    for images, labels in t_loader:
        images, labels = (
            images.to(device),
            labels.to(
                device,
                dtype=torch.float32
            )
                .unsqueeze(1)
        )

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(t_loader)

    print(f"[Epoch `{epoch+1}/{num_epochs}`] Loss: `{avg_loss:.04f}`")



    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in v_loader:
            images, labels = (
                images.to(device),
                labels.to(
                    device,
                    dtype=torch.float32,
                )
                    .unsqueeze(1)
            )

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(v_loader)

    print(f"Validation Loss: {val_loss:.4f}")



# Save model
filename = f"trained_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.pth"
torch.save(model.state_dict(), filename)
print(f"Model saved as `{filename}`")
