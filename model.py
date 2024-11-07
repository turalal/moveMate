import os
import pandas as pd
import numpy as np
import pydicom
import cv2
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm

class Config:
    # Paths
    TRAIN_DIR = '../input/train_images'
    TEST_DIR = '../input/test_images'
    
    # Training parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 5
    BATCH_SIZE = 8
    IMAGE_SIZE = 1024
    NUM_WORKERS = 4
    SEED = 42
    
    # Model parameters
    MODEL_NAME = 'efficientnet_b3'
    PRETRAINED = True
    NUM_CLASSES = 1
    
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
class MammographyDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, train=True):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, str(row.patient_id), f"{row.image_id}.dcm")
        
        # Read DICOM image
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array
        
        # Basic image preprocessing
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        if self.train:
            label = torch.tensor(row.cancer, dtype=torch.float)
            return image, label
        else:
            return image
        
class MammographyModel(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=1):
        super().__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes,
            in_chans=1
        )
        
    def forward(self, x):
        return self.model(x)
    
def get_transforms(image_size):
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])
    
    valid_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2(),
    ])
    
    return train_transform, valid_transform

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            running_loss += loss.item()
            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
    return running_loss / len(loader), predictions, targets

def main():
    # Set seed for reproducibility
    seed_everything(Config.SEED)
    
    # Read data
    train_df = pd.read_csv('../input/train.csv')
    
    # Create folds
    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=train_df.patient_id)):
        train_df.loc[val_idx, 'fold'] = fold
    
    # Training for one fold
    fold = 0
    
    # Create datasets and dataloaders
    train_transform, valid_transform = get_transforms(Config.IMAGE_SIZE)
    
    train_dataset = MammographyDataset(
        train_df[train_df.fold != fold],
        Config.TRAIN_DIR,
        transform=train_transform
    )
    
    valid_dataset = MammographyDataset(
        train_df[train_df.fold == fold],
        Config.TRAIN_DIR,
        transform=valid_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    # Create model
    model = MammographyModel(
        Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        print(f'Epoch {epoch + 1}/{Config.EPOCHS}')
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )
        
        val_loss, predictions, targets = validate(
            model, valid_loader, criterion, Config.DEVICE
        )
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Valid Loss: {val_loss:.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'model_fold_{fold}.pth')
            
def make_predictions(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
            
    return predictions

if __name__ == "__main__":
    main()