import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imgaug import augmenters as iaa
import torch.nn.functional as F


# ----------- Step 1: Labeling -----------

# Path to your local folder containing B3 and B4 images
dir_path = "/Users/doudou.ba/Downloads/niokolokoba"

print("Starting the labeling process...")

b3_files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if 'B3' in f])
b4_files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if 'B4' in f])

def extract_patches(image, patch_size=50, stride=25, threshold=0.5):
    forest_patches = []
    non_forest_patches = []
    for i in range(0, image.shape[0] - patch_size, stride):
        for j in range(0, image.shape[1] - patch_size, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            mean_ndvi = np.mean(patch)
            if mean_ndvi > threshold:
                forest_patches.append(patch)
            elif mean_ndvi < 0.2:
                non_forest_patches.append(patch)
    return forest_patches, non_forest_patches

all_forest_patches = []
all_non_forest_patches = []
for b3_path, b4_path in zip(b3_files, b4_files):
    with Image.open(b4_path) as nir_src:
        nir_data = np.array(nir_src)
    with Image.open(b3_path) as red_src:
        red_data = np.array(red_src)
    ndvi_data = (nir_data - red_data) / (nir_data + red_data + 1e-8)
    
    forest_patches, non_forest_patches = extract_patches(ndvi_data)
    all_forest_patches.extend(forest_patches)
    all_non_forest_patches.extend(non_forest_patches)

print("Labeling process completed.")

# ----------- Step 2: Data Augmentation -----------

print("Starting data augmentation...")

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rot90((1, 3))
])

forest_patches_augmented = seq(images=all_forest_patches)
non_forest_patches_augmented = seq(images=all_non_forest_patches)

X = forest_patches_augmented + non_forest_patches_augmented
y = [1]*len(forest_patches_augmented) + [0]*len(non_forest_patches_augmented)

# Convert PIL Images to numpy arrays, then stack
X_train = np.stack([np.array(img) for img in X])
y_train = np.array(y)

print("Data augmentation completed.")

# ----------- Step 3: Model Training -----------

print("Starting model training...")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 24 * 24, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 24 * 24)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

simple_model = SimpleCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(simple_model.parameters(), lr=0.001)

train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

for epoch in range(10):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = simple_model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

print("Model training completed.")
