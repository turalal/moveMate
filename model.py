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
import warnings
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
    BATCH_SIZE = 4  # Reduced batch size due to image size
    IMAGE_SIZE = 1024
    NUM_WORKERS = 2
    SEED = 42
    
    # Model parameters
    MODEL_NAME = 'efficientnet_b3'
    PRETRAINED = True
    NUM_CLASSES = 1
    LEARNING_RATE = 1e-4
    
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

class MammographyDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, train=True):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def load_dicom(self, path):
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array
        
        # Handle different bit depths
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
        # Ensure image has correct dimensions
        if len(img.shape) > 2:
            img = img[:, :, 0]  # Take first channel if multiple channels exist
            
        return img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, str(row.patient_id), f"{row.image_id}.dcm")
        
        try:
            image = self.load_dicom(image_path)
            
            # Apply transforms
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
                
            if self.train:
                label = torch.tensor(row.cancer, dtype=torch.float)
                return image, label
            else:
                return image
                
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return a blank image in case of error
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            if self.train:
                return image, torch.tensor(0, dtype=torch.float)
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
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.GridDistortion(p=0.3),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])
    
    valid_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])
    
    return train_transform, valid_transform

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc='Training')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    pbar = tqdm(loader, desc='Validating')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            running_loss += loss.item()
            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
    return running_loss / len(loader), predictions, targets

def main():
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
        pin_memory=True
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
        
        # Calculate metrics
        predictions = np.array(predictions) > 0.5
        targets = np.array(targets)
        accuracy = (predictions == targets).mean()
        print(f'Validation Accuracy: {accuracy:.4f}')
        
        # Save model if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            model_path = os.path.join(Config.OUTPUT_DIR, f'model_fold_{fold}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_path)
            print(f'Model saved to {model_path}')

if __name__ == "__main__":
    main()