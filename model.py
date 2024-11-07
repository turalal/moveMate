# First, install required dependencies
!pip install -q pylibjpeg pylibjpeg-libjpeg gdcm
!pip install -q pydicom==2.3.1
!pip install -q timm albumentations

import os
import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm
import warnings
import torch.cuda.amp as amp
warnings.filterwarnings('ignore')

class Config:
    # Kaggle paths
    TRAIN_CSV = '/kaggle/input/rsna-breast-cancer-detection/train.csv'
    TRAIN_DIR = '/kaggle/input/rsna-breast-cancer-detection/train_images'
    TEST_DIR = '/kaggle/input/rsna-breast-cancer-detection/test_images'
    OUTPUT_DIR = '/kaggle/working/'
    
    # Training parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 5
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    IMAGE_SIZE = 1024
    NUM_WORKERS = 2
    SEED = 42
    
    # Model parameters
    MODEL_NAME = 'efficientnet_b3'
    PRETRAINED = True
    NUM_CLASSES = 1
    LEARNING_RATE = 1e-4
    
    # Mixed precision training
    USE_AMP = True

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def prepare_dataframe(df):
    """Prepare the dataframe for training"""
    # Convert categorical columns to numeric
    df['laterality'] = df['laterality'].map({'L': 0, 'R': 1})
    df['view'] = df['view'].map({'CC': 0, 'MLO': 1})
    
    # Fill missing values
    df['age'].fillna(df['age'].mean(), inplace=True)
    df['implant'].fillna(0, inplace=True)
    df['density'].fillna(df['density'].mode()[0], inplace=True)
    
    # Convert density to numeric
    density_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    df['density'] = df['density'].map(density_map)
    
    return df

def read_dicom(path):
    """Read DICOM image and preprocess it properly"""
    try:
        dicom = pydicom.dcmread(path)
        
        # VOI LUT transformation
        if hasattr(dicom, 'WindowCenter'):
            voi_lut = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            voi_lut = dicom.pixel_array

        # Normalize to 8-bit
        if voi_lut.dtype != np.uint8:
            if voi_lut.max() < 256:
                img_8bit = voi_lut.astype(np.uint8)
            else:
                img_8bit = ((voi_lut - voi_lut.min()) / (voi_lut.max() - voi_lut.min()) * 255).astype(np.uint8)
        else:
            img_8bit = voi_lut

        return img_8bit
    
    except Exception as e:
        print(f"Error reading {path}: {str(e)}")
        return None

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
        
        image = read_dicom(image_path)
        
        if image is None:
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)
        elif len(image.shape) > 2:
            image = image[:, :, 0]
        
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
        A.ShiftScaleRotate(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(),
            A.GridDistortion(),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])
    
    valid_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])
    
    return train_transform, valid_transform

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, scaler=None):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc='Training')
    
    for step, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        with amp.autocast(enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
        
        if Config.USE_AMP:
            scaler.scale(loss).backward()
            if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        else:
            loss.backward()
            if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        
        running_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    pbar = tqdm(loader, desc='Validating')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        with amp.autocast(enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
        
        running_loss += loss.item()
        predictions.extend(torch.sigmoid(outputs).cpu().numpy())
        targets.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    auc = roc_auc_score(targets, predictions)
    
    return running_loss / len(loader), predictions, targets, auc

def main():
    print("Starting training pipeline...")
    
    # Set seed for reproducibility
    seed_everything(Config.SEED)
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Read and prepare data
    print("Loading and preparing data...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    train_df = prepare_dataframe(train_df)
    
    # Create folds
    gkf = GroupKFold(n_splits=5)
    train_df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=train_df.patient_id)):
        train_df.loc[val_idx, 'fold'] = fold
    
    # Training for one fold
    fold = 0
    print(f"Training fold {fold}")
    
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
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = MammographyModel(
        Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * Config.EPOCHS // Config.GRADIENT_ACCUMULATION_STEPS
    )
    
    # Initialize AMP scaler
    scaler = amp.GradScaler() if Config.USE_AMP else None
    
    # Training loop
    best_auc = 0
    
    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        print(f'Epoch {epoch + 1}/{Config.EPOCHS}')
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, 
            Config.DEVICE, scheduler, scaler
        )
        
        val_loss, predictions, targets, auc = validate(
            model, valid_loader, criterion, Config.DEVICE
        )
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Valid Loss: {val_loss:.4f}')
        print(f'Valid AUC: {auc:.4f}')
        
        # Save model if validation AUC improves
        if auc > best_auc:
            best_auc = auc
            model_path = os.path.join(Config.OUTPUT_DIR, f'model_fold_{fold}_auc_{auc:.4f}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'auc': best_auc,
            }, model_path)
            print(f'Model saved to {model_path}')

if __name__ == "__main__":
    main()