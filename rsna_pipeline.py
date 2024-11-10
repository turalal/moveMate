# Basic imports
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Core data processing
import numpy as np
import pandas as pd

# Machine learning
import sklearn
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

# Computer vision
import cv2
import pydicom
from pydicom.pixel_data_handlers import gdcm_handler, pillow_handler
pydicom.config.image_handlers = [gdcm_handler, pillow_handler]

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm

# Image augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Utilities
from pathlib import Path
from tqdm.auto import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

print("All imports successful!")

class CFG:
    """Configuration class containing all parameters"""
    # Debug mode
    debug = True  # Set to False for full training
    
    # Paths
    base_path = Path("/kaggle/input/rsna-breast-cancer-detection")
    processed_dir = Path("/kaggle/working/processed_images")
    model_dir = Path("/kaggle/working/models")
    
    # Preprocessing
    image_size = 512  # Single value for square images
    target_size = (512, 512)  # Reduced from 2048 for memory efficiency
    output_format = 'png'
    
    # Training parameters
    seed = 42
    epochs = 2 if debug else 10
    train_batch_size = 16
    valid_batch_size = 32
    num_workers = 4
    num_folds = 5
    
    # Model
    model_name = 'efficientnet_b3'
    pretrained = True
    num_classes = 1 
    
    # Optimizer
    optimizer = 'AdamW'
    learning_rate = 1e-4
    weight_decay = 1e-6
    
    # Scheduler
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-7
    T_max = int(epochs * 0.7)
    
    # Augmentations
    train_aug_list = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.GridDistortion(p=0.3),
        A.CoarseDropout(max_holes=8, max_width=20, max_height=20, p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ]
    
    valid_aug_list = [
        A.Normalize(),
        ToTensorV2(),
    ]
         
class RSNAPreprocessor:
    """Handles preprocessing of DICOM images"""
    def __init__(self, **kwargs):
        self.base_path = kwargs.get('base_path', CFG.base_path)
        self.target_size = kwargs.get('target_size', CFG.target_size)
        self.output_format = kwargs.get('output_format', CFG.output_format)
        
        self.train_images_path = self.base_path / "train_images"
        self.test_images_path = self.base_path / "test_images"
        
        if self.output_format not in ['png', 'jpg', 'jpeg']:
            raise ValueError("output_format must be 'png' or 'jpg'/'jpeg'")

    def read_dicom(self, patient_id, image_id, is_train=True):
        try:
            images_path = self.train_images_path if is_train else self.test_images_path
            dicom_path = images_path / str(patient_id) / f"{image_id}.dcm"
            
            if not dicom_path.exists():
                print(f"File not found: {dicom_path}")
                return None
            
            # Try multiple methods to read DICOM
            try:
                dicom = pydicom.dcmread(str(dicom_path), force=True)
                if hasattr(dicom, 'file_meta') and hasattr(dicom.file_meta, 'TransferSyntaxUID'):
                    if dicom.file_meta.TransferSyntaxUID.is_compressed:
                        dicom.decompress()
                img = dicom.pixel_array
            except Exception as e:
                print(f"Error reading DICOM with primary method: {str(e)}")
                return None
                
            # Convert to float and normalize
            img = img.astype(float)
            if img.max() != img.min():
                img = (img - img.min()) / (img.max() - img.min())
            img = (img * 255).astype(np.uint8)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            # Resize with padding
            img = self._resize_with_padding(img)
            
            return img
                
        except Exception as e:
            print(f"Error processing image {image_id} for patient {patient_id}: {str(e)}")
            return None

    def _try_alternate_reading(self, dicom_path):
        """Try different methods to read problematic DICOM files"""
        try:
            # Try GDCM first
            pydicom.config.image_handlers = [pydicom.pixel_data_handlers.gdcm_handler]
            dicom = pydicom.dcmread(str(dicom_path), force=True)
            return dicom
        except:
            try:
                # Try PyLibJPEG
                pydicom.config.image_handlers = [pydicom.pixel_data_handlers.pillow_handler]
                dicom = pydicom.dcmread(str(dicom_path), force=True)
                return dicom
            except:
                try:
                    # Try without any specific handler
                    pydicom.config.image_handlers = [None]
                    dicom = pydicom.dcmread(str(dicom_path), force=True)
                    return dicom
                except:
                    return None

    def _process_dicom_image(self, dicom):
        try:
            img = dicom.pixel_array.astype(np.float32)
            
            # Basic normalization
            if img.max() != img.min():
                img = (img - img.min()) / (img.max() - img.min())
            img = (img * 255).astype(np.uint8)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            return img
                
        except Exception as e:
            print(f"Error in _process_dicom_image: {str(e)}")
            return None

    def _resize_with_padding(self, img):
        if img is None:
            return None
            
        aspect = img.shape[0] / img.shape[1]
        if aspect > 1:
            new_height = self.target_size[0]
            new_width = int(new_height / aspect)
        else:
            new_width = self.target_size[1]
            new_height = int(new_width * aspect)
        
        img = cv2.resize(img, (new_width, new_height))
        
        # Add padding
        top_pad = (self.target_size[0] - img.shape[0]) // 2
        bottom_pad = self.target_size[0] - img.shape[0] - top_pad
        left_pad = (self.target_size[1] - img.shape[1]) // 2
        right_pad = self.target_size[1] - img.shape[1] - left_pad
        
        return cv2.copyMakeBorder(
            img, top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT, value=0
        )

    def save_image(self, img, output_path):
        """Save image with error checking"""
        try:
            if img is not None and img.size > 0:
                if self.output_format == 'png':
                    success = cv2.imwrite(str(output_path.with_suffix('.png')), img)
                else:
                    success = cv2.imwrite(str(output_path.with_suffix('.jpg')), img, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 100])
                return success
            return False
        except Exception as e:
            print(f"Error saving image to {output_path}: {str(e)}")
            return False
    def process_and_save(self, metadata_df, output_dir, num_samples=None):
        if num_samples:
            metadata_df = metadata_df.head(num_samples)
        
        output_dir = Path(output_dir)
        self._create_directory_structure(output_dir)
        
        processed_count = 0
        failed_count = 0
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            try:
                img = self.read_dicom(
                    patient_id=str(row['patient_id']),
                    image_id=str(row['image_id'])
                )
                
                if img is not None and img.size > 0:  # Add size check
                    # Save main image
                    output_path = output_dir / row['view'] / row['laterality'] / f"{row['patient_id']}_{row['image_id']}"
                    success = self.save_image(img, output_path)
                    
                    if success:
                        # Only save thumbnail if main image was saved successfully
                        thumbnail = cv2.resize(img, (512, 512))
                        thumbnail_path = output_path.with_name(f"{output_path.stem}_thumb")
                        self.save_image(thumbnail, thumbnail_path)
                        processed_count += 1
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                print(f"Error processing row {idx}: {str(e)}")
        
        return processed_count, failed_count

    def _create_directory_structure(self, output_dir):
        output_dir.mkdir(exist_ok=True)
        for view in ['CC', 'MLO']:
            (output_dir / view).mkdir(exist_ok=True)
            (output_dir / view / 'L').mkdir(exist_ok=True)
            (output_dir / view / 'R').mkdir(exist_ok=True)

class RSNADataset(Dataset):
    """PyTorch Dataset for RSNA images"""
    def __init__(self, df, transform=None, is_train=True):
        self.df = df
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = CFG.processed_dir / row['view'] / row['laterality'] / f"{row['patient_id']}_{row['image_id']}.png"
        
        if not img_path.exists():
            # Create blank image using tuple target_size
            img = np.zeros((CFG.target_size[0], CFG.target_size[1], 3), dtype=np.uint8)
            print(f"Warning: Image not found: {img_path}")
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((CFG.target_size[0], CFG.target_size[1], 3), dtype=np.uint8)
                print(f"Warning: Failed to load image: {img_path}")
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(image=img)['image']
            
        if self.is_train:
            label = torch.tensor(row['cancer'], dtype=torch.float32)
            return img, label
        else:
            return img


class RSNAModel(nn.Module):
    """Model architecture"""
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            CFG.model_name, 
            pretrained=CFG.pretrained, 
            num_classes=CFG.num_classes  # Changed from target_size
        )
        
    def forward(self, x):
        return self.model(x)
    
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """Trains the model for one epoch"""
    model.train()
    scaler = GradScaler()
    losses = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        with autocast():
            y_preds = model(images).squeeze(1)
            loss = criterion(y_preds, labels)
        
        losses.update(loss.item(), labels.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
            
        pbar.set_postfix({'train_loss': losses.avg})
    
    return losses.avg

def valid_one_epoch(model, valid_loader, criterion, device):
    """Validates the model for one epoch with safe metric calculation"""
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []
    
    pbar = tqdm(valid_loader, desc='Validation')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            y_preds = model(images).squeeze(1)
            loss = criterion(y_preds, labels)
            
            losses.update(loss.item(), labels.size(0))
            preds.append(y_preds.sigmoid().cpu().numpy())
            targets.append(labels.cpu().numpy())
            
            pbar.set_postfix({'valid_loss': losses.avg})
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    # Safe ROC AUC calculation
    try:
        if len(np.unique(targets)) > 1:
            score = roc_auc_score(targets, preds)
        else:
            print("Warning: Only one class present in validation set. Using loss as score.")
            score = -losses.avg  # Use negative loss as score
    except Exception as e:
        print(f"Error calculating score: {str(e)}")
        score = -losses.avg
    
    return losses.avg, score

def get_dataloader(dataset, batch_size, shuffle=True, is_train=True):
    """Safe DataLoader creation with proper worker initialization"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
        drop_last=is_train
    )

def train_model():
    """Main training loop with stratified fold splitting"""
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    
    # Read and prepare data
    train_df = pd.read_csv(CFG.base_path / 'train.csv')
    if CFG.debug:
        train_df = train_df.head(100)
    
    # Print class distribution
    print("\nClass distribution in training data:")
    print(train_df['cancer'].value_counts(normalize=True))
    
    # Create stratified folds
    skf = StratifiedGroupKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)
    train_df['fold'] = -1  # Initialize fold column
    
    # Assign folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['cancer'], groups=train_df['patient_id'])):
        train_df.loc[val_idx, 'fold'] = fold
    
    # Training loop
    for fold in range(CFG.num_folds):
        print(f'\nTraining fold {fold + 1}/{CFG.num_folds}')
        
        # Clear GPU memory before each fold
        torch.cuda.empty_cache()
        gc.collect()
        
        # Check class distribution in fold
        train_fold = train_df[train_df.fold != fold].reset_index(drop=True)
        valid_fold = train_df[train_df.fold == fold].reset_index(drop=True)
        
        print("\nTrain fold class distribution:")
        print(train_fold['cancer'].value_counts(normalize=True))
        print("\nValid fold class distribution:")
        print(valid_fold['cancer'].value_counts(normalize=True))
        
        try:
            # Prepare datasets
            train_dataset = RSNADataset(train_fold, transform=A.Compose(CFG.train_aug_list))
            valid_dataset = RSNADataset(valid_fold, transform=A.Compose(CFG.valid_aug_list))
            
            # Create dataloaders with safe settings
            train_loader = get_dataloader(train_dataset, CFG.train_batch_size, shuffle=True)
            valid_loader = get_dataloader(valid_dataset, CFG.valid_batch_size, shuffle=False, is_train=False)
            
            # Initialize model and training
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = RSNAModel().to(device)
            
            # Calculate class weights for imbalanced dataset
            pos_weight = len(train_fold[train_fold['cancer'] == 0]) / max(1, len(train_fold[train_fold['cancer'] == 1]))
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
            
            optimizer = getattr(torch.optim, CFG.optimizer)(
                model.parameters(),
                lr=CFG.learning_rate,
                weight_decay=CFG.weight_decay
            )
            
            scheduler = getattr(torch.optim.lr_scheduler, CFG.scheduler)(
                optimizer,
                T_max=CFG.T_max,
                eta_min=CFG.min_lr
            )
            
            # Training
            best_score = float('-inf')
            patience = 3
            patience_counter = 0
            
            for epoch in range(CFG.epochs):
                print(f'Epoch {epoch + 1}/{CFG.epochs}')
                
                try:
                    train_loss = train_one_epoch(
                        model, train_loader, criterion,
                        optimizer, scheduler, device
                    )
                    
                    valid_loss, valid_score = valid_one_epoch(
                        model, valid_loader, criterion, device
                    )
                    
                    print(f'Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f} Valid Score: {valid_score:.4f}')
                    
                    # Save best model
                    if valid_score > best_score:
                        best_score = valid_score
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_score': best_score,
                            'epoch': epoch,
                        }, CFG.model_dir / f'fold{fold}_best.pth')
                        print(f'Best model saved! Score: {best_score:.4f}')
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    # Early stopping
                    if patience_counter >= patience:
                        print(f'Early stopping triggered after {epoch + 1} epochs')
                        break
                        
                except Exception as e:
                    print(f"Error in epoch {epoch + 1}: {str(e)}")
                    break
            
        except Exception as e:
            print(f"Error in fold {fold + 1}: {str(e)}")
            continue
            
        finally:
            # Cleanup
            try:
                del model, train_loader, valid_loader, train_dataset, valid_dataset
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass

        print(f"Completed fold {fold + 1} with best score: {best_score:.4f}")

    
def inference():
    """Performs inference using trained models"""
    print("\nStarting inference...")
    
    # Read test data
    test_df = pd.read_csv(CFG.base_path / 'test.csv')
    if CFG.debug:
        test_df = test_df.head(100)
        print("Debug mode: Using only 100 test samples")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = []
    
    # Create test dataset
    test_dataset = RSNADataset(
        test_df,
        transform=A.Compose(CFG.valid_aug_list),
        is_train=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True
    )
    
    # Inference with all folds
    for fold in range(CFG.num_folds):
        print(f'Inferencing fold {fold + 1}/{CFG.num_folds}')
        model = RSNAModel().to(device)
        
        try:
            model.load_state_dict(torch.load(CFG.model_dir / f'fold{fold}_best.pth'))
            model.eval()
            
            fold_preds = []
            with torch.no_grad():
                for images in tqdm(test_loader, desc=f'Fold {fold + 1}'):
                    images = images.to(device)
                    y_preds = model(images).squeeze(1)
                    fold_preds.append(y_preds.sigmoid().cpu().numpy())
            
            fold_preds = np.concatenate(fold_preds)
            predictions.append(fold_preds)
            
        except Exception as e:
            print(f"Error in fold {fold} inference: {str(e)}")
            continue
        
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    
    # Average predictions from all folds
    predictions = np.mean(predictions, axis=0)
    
    # Create submission
    submission = pd.DataFrame({
        'prediction_id': test_df['prediction_id'],
        'cancer': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print('Submission saved!')
    return submission

def process_test_data():
    """Processes test data for inference"""
    print("\nProcessing test data...")
    test_df = pd.read_csv(CFG.base_path / 'test.csv')
    if CFG.debug:
        test_df = test_df.head(100)
    
    preprocessor = RSNAPreprocessor(
        base_path=CFG.base_path,
        target_size=CFG.target_size,
        output_format=CFG.output_format
    )
    
    processed_count, failed_count = preprocessor.process_and_save(
        test_df,
        CFG.processed_dir,
        num_samples=None if not CFG.debug else 100
    )
    print(f"Test data processing completed. Processed: {processed_count}, Failed: {failed_count}")

def main():
    """Main execution function"""
    print("Starting RSNA Mammography Pipeline...")
    
    try:
        # Create necessary directories
        CFG.processed_dir.mkdir(parents=True, exist_ok=True)
        CFG.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Process training data
        print("\nStep 1: Processing training data...")
        train_df = pd.read_csv(CFG.base_path / 'train.csv')
        if CFG.debug:
            train_df = train_df.head(100)
            print("Debug mode: Using only 100 training samples")
        
        preprocessor = RSNAPreprocessor(
            base_path=CFG.base_path,
            target_size=CFG.target_size,
            output_format=CFG.output_format
        )
        
        processed_count, failed_count = preprocessor.process_and_save(
            train_df,
            CFG.processed_dir,
            num_samples=None if not CFG.debug else 100
        )
        print(f"Training data processing completed. Processed: {processed_count}, Failed: {failed_count}")
        
        # Step 2: Train models
        print("\nStep 2: Training models...")
        train_model()
        
        # Step 3: Process test data
        print("\nStep 3: Processing test data...")
        process_test_data()
        
        # Step 4: Generate predictions
        print("\nStep 4: Generating predictions...")
        submission = inference()
        
        print("\nPipeline completed successfully!")
        
        if CFG.debug:
            print("\nNote: This was run in debug mode. Set CFG.debug = False for full training.")
        
        return submission
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()