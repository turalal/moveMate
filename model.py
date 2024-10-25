import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import convert_color_space
import torch_xla
import torch_xla.core.xla_model as xm

# Load CSV data
train_df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")
test_df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/test.csv")

# DICOM Dataset class
class RSNADataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['patient_id']
        image_id = row['image_id']
        
        # Create file path for DICOM images
        dicom_path = os.path.join(self.img_dir, str(patient_id), f"{image_id}.dcm")
        
        if not os.path.exists(dicom_path):
            print(f"File not found: {dicom_path}")
            return None, None
        
        # Read DICOM file using pydicom and handle compressed DICOMs
        try:
            dicom_img = pydicom.dcmread(dicom_path, force=True)
            dicom_img = dicom_img.pixel_array
        except Exception as e:
            print(f"Failed to read DICOM file: {dicom_path}, Error: {e}")
            return None, None
        
        # Convert image to torch tensor and cast to float32 for normalization
        dicom_img = torch.tensor(dicom_img, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

        if self.transform:
            dicom_img = self.transform(dicom_img)

        label = row['cancer'] if 'cancer' in row else -1  # No label for test set
        return dicom_img, label

# Define the transformation (resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets using the train and test DICOM image paths
train_images_dir = "/kaggle/input/rsna-breast-cancer-detection/train_images"
test_images_dir = "/kaggle/input/rsna-breast-cancer-detection/test_images"

train_dataset = RSNADataset(df=train_df, img_dir=train_images_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

# Simple model (using ResNet18 as an example)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model = resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Output: binary

    def forward(self, x):
        return self.model(x)

# Initialize model, loss, optimizer
device = xm.xla_device()  # Use TPU device
model = SimpleModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_one_epoch(loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader):
        if images is None:  # Skip batches with None
            continue
        images = images.to(device, dtype=torch.float32)  # Ensure input is float32
        labels = labels.to(device, dtype=torch.float32).unsqueeze(1)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)  # TPU optimizer step

        running_loss += loss.item()
    return running_loss / len(loader)

# Run a small batch prediction
def predict(loader, model):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in loader:
            if images is None:
                continue
            images = images.to(device, dtype=torch.float32)  # Ensure input is float32
            outputs = model(images)
            predictions.append(outputs.cpu().numpy())
    return predictions

# Training and prediction example
epochs = 1
for epoch in range(epochs):
    train_loss = train_one_epoch(train_loader, model, criterion, optimizer)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}")

# Test prediction on a small batch
test_dataset = RSNADataset(df=test_df, img_dir=test_images_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# Get predictions
predictions = predict(test_loader, model)
print(predictions)
