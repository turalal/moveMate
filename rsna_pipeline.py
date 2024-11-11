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
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight


# Computer vision
import cv2
import pydicom
from pydicom.pixel_data_handlers import gdcm_handler, pillow_handler
pydicom.config.image_handlers = [gdcm_handler, pillow_handler]

# Deep learning
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm

# Image augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Monitoring
import matplotlib.pyplot as plt
import traceback
from datetime import datetime

# Logging
import sys
import logging

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
    train_batch_size = 8
    valid_batch_size = 32
    num_workers = 0
    num_folds = 5
    patience = 3  # Added for early stopping
    # Add safety flags
    persistent_workers = False
    pin_memory = True
    
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
    
    # Class balancing parameters
    focal_loss_alpha = 0.25
    focal_loss_gamma = 2.0
    use_class_weights = True
    oversample_minority = True
    
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

class BalancedRSNADataset(Dataset):  # Changed to inherit from Dataset
    def __init__(self, df, transform=None, is_train=True):
        self.df = df
        self.transform = transform
        self.is_train = is_train
        self.image_cache = {}  # Add image caching
        
        # Calculate class weights if training
        if is_train:
            self.class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(df['cancer']),
                y=df['cancer']
            )
            self.class_weights = torch.FloatTensor(self.class_weights)
            
            # Calculate sample weights for WeightedRandomSampler
            self.sample_weights = [
                self.class_weights[int(label)] for label in df['cancer']
            ]
    
    def _load_image(self, img_path):
        """Load image with caching and error handling"""
        if img_path in self.image_cache:
            return self.image_cache[img_path].copy()
            
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Cache image if not in training mode (to save memory)
                if not self.is_train:
                    self.image_cache[img_path] = img.copy()
                return img
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
        
        return None
    
    def _get_blank_image(self):
        """Create blank image with proper dimensions"""
        return np.zeros((CFG.target_size[0], CFG.target_size[1], 3), dtype=np.uint8)
             
    def get_sampler(self):
        """Returns WeightedRandomSampler for balanced batches"""
        if self.is_train:
            return WeightedRandomSampler(
                self.sample_weights,
                len(self.sample_weights),
                replacement=True
            )
        return None
    
    def __getitem__(self, idx):
        # Same as your original RSNADataset __getitem__
        row = self.df.iloc[idx]
        img_path = CFG.processed_dir / row['view'] / row['laterality'] / f"{row['patient_id']}_{row['image_id']}.png"
        
        img = self._load_image(img_path)
        if img is None:
            img = self._get_blank_image()
            print(f"Warning: Using blank image for: {img_path}")
        
        if self.transform:
            try:
                transformed = self.transform(image=img)
                img = transformed['image']
            except Exception as e:
                print(f"Error in transformation: {str(e)}")
                img = self.transform(image=self._get_blank_image())['image']
        
        if self.is_train:
            label = torch.tensor(row['cancer'], dtype=torch.float32)
            return img, label
        else:
            return img
    
    def __len__(self):
        return len(self.df)

    def get_class_distribution(self):
        """Calculate current class distribution"""
        return self.df['cancer'].value_counts(normalize=True).to_dict()

    def get_class_ratio(self):
        """Calculate ratio between classes"""
        dist = self.get_class_distribution()
        return dist[0] / dist[1] if 1 in dist else float('inf')

    def get_all_images(self):
        """Get all images as a tensor"""
        images = []
        for idx in range(len(self)):
            if self.is_train:
                img, _ = self[idx]
            else:
                img = self[idx]
            images.append(img)
        return torch.stack(images)
    
    def get_all_labels(self):
        """Get all labels as a tensor"""
        if not self.is_train:
            return None
        return torch.tensor(self.df['cancer'].values, dtype=torch.float32)

class FocalLoss(nn.Module):
    """Focal Loss for dealing with class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
      
class RSNAPreprocessor:
    """Handles preprocessing of DICOM images"""
    """Enhanced preprocessing with parallel processing and better error handling"""
    def __init__(self, **kwargs):
        self.base_path = kwargs.get('base_path', CFG.base_path)
        self.target_size = kwargs.get('target_size', CFG.target_size)
        self.output_format = kwargs.get('output_format', CFG.output_format)
        
        # Initialize paths
        self.train_images_path = self.base_path / "train_images"
        self.test_images_path = self.base_path / "test_images"
        
        # Validate output format
        if self.output_format not in ['png', 'jpg', 'jpeg']:
            raise ValueError("output_format must be 'png' or 'jpg'/'jpeg'")
        
        # Initialize error logging
        self.error_log = []
        
        # Initialize CLAHE object once
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Verify paths exist
        if not self.train_images_path.exists():
            raise ValueError(f"Train images path does not exist: {self.train_images_path}")
        if not self.test_images_path.exists():
            raise ValueError(f"Test images path does not exist: {self.test_images_path}")
            
        # Set image handlers for pydicom
        pydicom.config.image_handlers = [gdcm_handler, pillow_handler]

    def read_dicom(self, patient_id, image_id, is_train=True):
        """Enhanced DICOM reading with better error handling and performance"""
        try:
            images_path = self.train_images_path if is_train else self.test_images_path
            dicom_path = images_path / str(patient_id) / f"{image_id}.dcm"
            
            if not dicom_path.exists():
                error_msg = f"File not found: {dicom_path}"
                self.error_log.append({
                    'patient_id': patient_id,
                    'image_id': image_id,
                    'error': error_msg
                })
                print(error_msg)
                return None
            
            try:
                # Use primary reading method
                dicom = pydicom.dcmread(str(dicom_path), force=True)
                if hasattr(dicom, 'file_meta') and hasattr(dicom.file_meta, 'TransferSyntaxUID'):
                    if dicom.file_meta.TransferSyntaxUID.is_compressed:
                        dicom.decompress()
                img = dicom.pixel_array
                
            except Exception as e:
                # Try alternate reading methods if primary fails
                dicom = self._try_alternate_reading(dicom_path)
                if dicom is None:
                    error_msg = f"Failed to read DICOM with all methods: {str(e)}"
                    self.error_log.append({
                        'patient_id': patient_id,
                        'image_id': image_id,
                        'error': error_msg
                    })
                    print(error_msg)
                    return None
                img = dicom.pixel_array
                
            # Convert to float and normalize
            img = img.astype(float)
            if img.max() != img.min():
                img = (img - img.min()) / (img.max() - img.min())
            img = (img * 255).astype(np.uint8)
            
            # Apply CLAHE for better contrast - use instance variable
            img = self.clahe.apply(img)
            
            # Resize with padding
            img = self._resize_with_padding(img)
            
            return img
                
        except Exception as e:
            error_msg = f"Error processing image {image_id} for patient {patient_id}: {str(e)}"
            self.error_log.append({
                'patient_id': patient_id,
                'image_id': image_id,
                'error': error_msg
            })
            print(error_msg)
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
        """Enhanced DICOM processing with better error handling"""
        try:
            img = dicom.pixel_array.astype(np.float32)
            
            # Basic normalization
            if img.max() != img.min():
                img = (img - img.min()) / (img.max() - img.min())
            img = (img * 255).astype(np.uint8)
            
            # Use instance CLAHE object
            img = self.clahe.apply(img)
            
            return img
                
        except Exception as e:
            print(f"Error in _process_dicom_image: {str(e)}")
            return None

    def _resize_with_padding(self, img):
        """Enhanced resize with better error checking"""
        if img is None:
            return None
            
        try:
            aspect = img.shape[0] / img.shape[1]
            if aspect > 1:
                new_height = self.target_size[0]
                new_width = int(new_height / aspect)
            else:
                new_width = self.target_size[1]
                new_height = int(new_width * aspect)
            
            # Use INTER_AREA for downscaling, INTER_LINEAR for upscaling
            if img.shape[0] > new_height or img.shape[1] > new_width:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_LINEAR
                
            img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
            
            # Add padding
            top_pad = (self.target_size[0] - img.shape[0]) // 2
            bottom_pad = self.target_size[0] - img.shape[0] - top_pad
            left_pad = (self.target_size[1] - img.shape[1]) // 2
            right_pad = self.target_size[1] - img.shape[1] - left_pad
            
            return cv2.copyMakeBorder(
                img, top_pad, bottom_pad, left_pad, right_pad,
                cv2.BORDER_CONSTANT, value=0
            )
        except Exception as e:
            print(f"Error in resize_with_padding: {str(e)}")
            return None

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
     
    def _validate_dicom(self, dicom, patient_id, image_id):
        """Validates DICOM file and logs metadata"""
        try:
            # Required DICOM attributes for validation
            required_attributes = [
                'PatientID', 
                'StudyInstanceUID', 
                'SeriesInstanceUID',
                'Rows', 
                'Columns'
            ]
            
            # Check for required attributes
            missing_attributes = [
                attr for attr in required_attributes 
                if not hasattr(dicom, attr)
            ]
            
            if missing_attributes:
                self.error_log.append({
                    'patient_id': patient_id,
                    'image_id': image_id,
                    'error': f'Missing required DICOM attributes: {missing_attributes}',
                    'type': 'validation_error'
                })
                return False
                
            # Validate image dimensions
            if dicom.Rows == 0 or dicom.Columns == 0:
                self.error_log.append({
                    'patient_id': patient_id,
                    'image_id': image_id,
                    'error': f'Invalid image dimensions: {dicom.Rows}x{dicom.Columns}',
                    'type': 'dimension_error'
                })
                return False
                
            # Log metadata for analysis
            metadata = {
                'patient_id': patient_id,
                'image_id': image_id,
                'rows': dicom.Rows,
                'columns': dicom.Columns,
                'bits_allocated': getattr(dicom, 'BitsAllocated', None),
                'bits_stored': getattr(dicom, 'BitsStored', None),
                'pixel_representation': getattr(dicom, 'PixelRepresentation', None),
                'window_center': getattr(dicom, 'WindowCenter', None),
                'window_width': getattr(dicom, 'WindowWidth', None),
                'modality': getattr(dicom, 'Modality', None)
            }
            
            # Save metadata
            if not hasattr(self, 'metadata_log'):
                self.metadata_log = []
            self.metadata_log.append(metadata)
            
            return True
            
        except Exception as e:
            self.error_log.append({
                'patient_id': patient_id,
                'image_id': image_id,
                'error': str(e),
                'type': 'validation_exception'
            })
            return False

    def process_and_save(self, metadata_df, output_dir, num_samples=None):
        """Enhanced data processing and saving with detailed logging"""
        try:
            if num_samples:
                metadata_df = metadata_df.head(num_samples)
            
            output_dir = Path(output_dir)
            self._create_directory_structure(output_dir)
            
            # Initialize counters and logs
            processed_count = 0
            failed_count = 0
            batch_stats = []
            
            # Process in smaller batches to manage memory
            batch_size = 50
            num_batches = (len(metadata_df) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(metadata_df))
                batch_df = metadata_df.iloc[start_idx:end_idx]
                
                batch_processed = 0
                batch_failed = 0
                batch_start_time = pd.Timestamp.now()
                
                # Process batch
                for idx, row in tqdm(batch_df.iterrows(), 
                                total=len(batch_df),
                                desc=f'Processing batch {batch_idx + 1}/{num_batches}'):
                    try:
                        # Read DICOM
                        img = self.read_dicom(
                            patient_id=str(row['patient_id']),
                            image_id=str(row['image_id'])
                        )
                        
                        if img is not None and img.size > 0:
                            # Prepare output paths
                            output_path = (output_dir / row['view'] / row['laterality'] / 
                                        f"{row['patient_id']}_{row['image_id']}")
                            
                            # Save main image
                            success = self.save_image(img, output_path)
                            
                            if success:
                                # Create and save thumbnail
                                try:
                                    thumbnail = cv2.resize(img, (512, 512))
                                    thumbnail_path = output_path.with_name(f"{output_path.stem}_thumb")
                                    thumb_success = self.save_image(thumbnail, thumbnail_path)
                                    
                                    if thumb_success:
                                        processed_count += 1
                                        batch_processed += 1
                                    else:
                                        failed_count += 1
                                        batch_failed += 1
                                        self.error_log.append({
                                            'patient_id': row['patient_id'],
                                            'image_id': row['image_id'],
                                            'error': 'Failed to save thumbnail',
                                            'batch': batch_idx + 1,
                                            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                                        })
                                except Exception as thumb_error:
                                    failed_count += 1
                                    batch_failed += 1
                                    self.error_log.append({
                                        'patient_id': row['patient_id'],
                                        'image_id': row['image_id'],
                                        'error': f'Thumbnail error: {str(thumb_error)}',
                                        'batch': batch_idx + 1,
                                        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                                    })
                            else:
                                failed_count += 1
                                batch_failed += 1
                                self.error_log.append({
                                    'patient_id': row['patient_id'],
                                    'image_id': row['image_id'],
                                    'error': 'Failed to save main image',
                                    'batch': batch_idx + 1,
                                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                                })
                        else:
                            failed_count += 1
                            batch_failed += 1
                            self.error_log.append({
                                'patient_id': row['patient_id'],
                                'image_id': row['image_id'],
                                'error': 'Invalid image data',
                                'batch': batch_idx + 1,
                                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                            
                    except Exception as e:
                        failed_count += 1
                        batch_failed += 1
                        self.error_log.append({
                            'patient_id': row['patient_id'],
                            'image_id': row['image_id'],
                            'error': str(e),
                            'batch': batch_idx + 1,
                            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        print(f"Error processing row {idx}: {str(e)}")
                
                # Record batch statistics
                batch_end_time = pd.Timestamp.now()
                batch_duration = (batch_end_time - batch_start_time).total_seconds()
                batch_stats.append({
                    'batch': batch_idx + 1,
                    'processed': batch_processed,
                    'failed': batch_failed,
                    'duration': batch_duration,
                    'start_time': batch_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': batch_end_time.strftime('%Y-%m-%d %H:%M:%S')
                })
                
                # Clear memory after each batch
                gc.collect()
            
            # Save logs and statistics
            try:
                # Save error log with additional details
                if self.error_log:
                    error_df = pd.DataFrame(self.error_log)
                    error_summary = error_df.groupby('error')['patient_id'].count()
                    
                    # Save detailed error log
                    error_df.to_csv(output_dir / 'processing_errors.csv', index=False)
                    
                    # Save error summary
                    with open(output_dir / 'error_summary.txt', 'w') as f:
                        f.write("Error Summary:\n")
                        f.write(str(error_summary))
                    
                    print(f"\nError log saved with {len(self.error_log)} entries")
                    print("\nError Summary:")
                    print(error_summary)
                
                # Save batch statistics
                if batch_stats:
                    stats_df = pd.DataFrame(batch_stats)
                    stats_df.to_csv(output_dir / 'batch_statistics.csv', index=False)
                    
                    # Calculate and save processing summary
                    total_duration = sum(stat['duration'] for stat in batch_stats)
                    avg_time_per_image = total_duration / (processed_count + failed_count)
                    
                    with open(output_dir / 'processing_summary.txt', 'w') as f:
                        f.write(f"Processing Summary:\n")
                        f.write(f"Total Images Processed: {processed_count}\n")
                        f.write(f"Total Images Failed: {failed_count}\n")
                        f.write(f"Total Processing Time: {total_duration:.2f} seconds\n")
                        f.write(f"Average Time per Image: {avg_time_per_image:.2f} seconds\n")
                        f.write(f"Success Rate: {(processed_count / (processed_count + failed_count) * 100):.2f}%\n")
                
                # Save DICOM metadata if collected
                if hasattr(self, 'metadata_log') and self.metadata_log:
                    metadata_df = pd.DataFrame(self.metadata_log)
                    metadata_df.to_csv(output_dir / 'dicom_metadata.csv', index=False)
                    print(f"\nDICOM metadata saved for {len(self.metadata_log)} files")
                    
            except Exception as log_error:
                print(f"Error saving logs: {str(log_error)}")
            
            return processed_count, failed_count
            
        except Exception as e:
            print(f"Fatal error in process_and_save: {str(e)}")
            return 0, 0

    def _create_directory_structure(self, output_dir):
        output_dir.mkdir(exist_ok=True)
        for view in ['CC', 'MLO']:
            (output_dir / view).mkdir(exist_ok=True)
            (output_dir / view / 'L').mkdir(exist_ok=True)
            (output_dir / view / 'R').mkdir(exist_ok=True)

class RSNADataset(Dataset):
    """Enhanced PyTorch Dataset for RSNA images"""
    def __init__(self, df, transform=None, is_train=True):
        self.df = df
        self.transform = transform
        self.is_train = is_train
        self.image_cache = {}  # Add image caching
        
    def __len__(self):
        return len(self.df)
    
    def _load_image(self, img_path):
        """Load image with caching and error handling"""
        if img_path in self.image_cache:
            return self.image_cache[img_path].copy()
            
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Cache image if not in training mode (to save memory)
                if not self.is_train:
                    self.image_cache[img_path] = img.copy()
                return img
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
        
        return None
    
    def _get_blank_image(self):
        """Create blank image with proper dimensions"""
        return np.zeros((CFG.target_size[0], CFG.target_size[1], 3), dtype=np.uint8)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = CFG.processed_dir / row['view'] / row['laterality'] / f"{row['patient_id']}_{row['image_id']}.png"
        
        # Load image or get blank if failed
        img = self._load_image(img_path)
        if img is None:
            img = self._get_blank_image()
            print(f"Warning: Using blank image for: {img_path}")
        
        # Apply augmentations
        if self.transform:
            try:
                transformed = self.transform(image=img)
                img = transformed['image']
            except Exception as e:
                print(f"Error in transformation: {str(e)}")
                img = self.transform(image=self._get_blank_image())['image']
        
        if self.is_train:
            label = torch.tensor(row['cancer'], dtype=torch.float32)
            return img, label
        else:
            return img
    
    def clear_cache(self):
        """Clear the image cache"""
        self.image_cache.clear()

class RSNAModel(nn.Module):
    """Enhanced model architecture with attention and feature extraction"""
    def __init__(self, model_name=CFG.model_name, pretrained=CFG.pretrained):
        super().__init__()
        
        # Initialize base model
        self.base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classifier to add custom head
        )
        
        # Get number of features from base model
        self.num_features = self.base_model.num_features
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 16),
            nn.ReLU(),
            nn.Linear(self.num_features // 16, self.num_features),
            nn.Sigmoid()
        )
        
        # Add classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_features, CFG.num_classes)
        )
        
        # Initialize metrics tracking
        self.batch_predictions = []
        self.batch_targets = []
        
    def forward(self, x):
        """Forward pass with attention mechanism"""
        # Get features from base model
        features = self.base_model.forward_features(x)
        
        # Apply global average pooling if needed
        if len(features.shape) > 2:
            features = nn.functional.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Final classification
        output = self.classifier(features)
        return output
    
    def get_attention_maps(self, x):
        """Get attention maps for visualization"""
        self.eval()
        with torch.no_grad():
            features = self.base_model.forward_features(x)
            attention = self.attention(features.mean((-2, -1)))
            attention = attention.view(attention.size(0), -1, 1, 1)
            attention_maps = features * attention
        return attention_maps
    
    def get_features(self, x):
        """Extract features for analysis"""
        self.eval()
        with torch.no_grad():
            features = self.base_model.forward_features(x)
            if len(features.shape) > 2:
                features = nn.functional.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        return features
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_info(self):
        """Get information about model layers"""
        layers_info = []
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                # num_params = sum(p.numel() for p in module.parameters(if_exists=True))
                num_params = sum(p.numel() for p in module.parameters())
                layers_info.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': num_params,
                })
        return pd.DataFrame(layers_info)
    
    @torch.no_grad()
    def update_metrics(self, outputs, targets):
        """Update batch-wise metrics"""
        predictions = torch.sigmoid(outputs).cpu().numpy()
        targets = targets.cpu().numpy()
        
        self.batch_predictions.extend(predictions)
        self.batch_targets.extend(targets)
    
    def get_metrics(self):
        """Calculate current metrics"""
        if not self.batch_predictions:
            return {}
            
        predictions = np.array(self.batch_predictions)
        targets = np.array(self.batch_targets)
        
        try:
            auc_score = roc_auc_score(targets, predictions)
        except:
            auc_score = float('nan')
            
        metrics = {
            'auc_score': auc_score,
            'avg_prediction': predictions.mean(),
            'pos_ratio': (targets == 1).mean(),
        }
        
        # Reset tracking
        self.batch_predictions = []
        self.batch_targets = []
        
        return metrics

    def freeze_backbone(self, freeze=True):
        """Freeze/unfreeze backbone for transfer learning"""
        for param in self.base_model.parameters():
            param.requires_grad = not freeze
            
    def load_pretrained(self, checkpoint_path):
        """Safe loading of pretrained weights"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
            return True
        except Exception as e:
            print(f"Error loading pretrained weights: {str(e)}")
            return False
        
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

class TrainingMonitor:
    """Training monitoring and visualization utilities"""
    def __init__(self, cfg):
        self.cfg = cfg
        
        # Import required libraries
        import matplotlib.pyplot as plt
        self.plt = plt
        
        # Create directories for saving monitoring data
        self.monitor_dir = cfg.model_dir / 'monitoring'
        self.monitor_dir.mkdir(exist_ok=True)
        
    def visualize_training_progress(self, fold_history):
        """Plot training progress for a fold"""
        epochs = range(1, len(fold_history['train_loss']) + 1)
        
        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(epochs, fold_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, fold_history['valid_loss'], 'r-', label='Valid Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot validation score
        ax2.plot(epochs, fold_history['valid_score'], 'g-', label='Valid Score')
        ax2.set_title('Validation Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True)
        
        self.plt.tight_layout()
        return fig
    
    def print_model_summary(self, model):
        """Print detailed model summary"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Summary:")
        print(f"Architecture: {self.cfg.model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Layer-wise summary
        layer_info = model.get_layer_info()
        print("\nLayer-wise parameter distribution:")
        for _, row in layer_info.iterrows():
            if row['parameters'] > 0:
                print(f"{row['name']}: {row['type']} - {row['parameters']:,} parameters")
    
    def log_fold_metrics(self, fold, fold_history, best_score, best_loss):
        """Log fold metrics to file"""
        metrics_path = self.monitor_dir / 'training_metrics.txt'
        
        with open(metrics_path, 'a') as f:
            f.write(f"\nFold {fold + 1} Results:\n")
            f.write(f"Best Score: {best_score:.4f}\n")
            f.write(f"Best Loss: {best_loss:.4f}\n")
            f.write(f"Final Train Loss: {fold_history['train_loss'][-1]:.4f}\n")
            f.write(f"Final Valid Loss: {fold_history['valid_loss'][-1]:.4f}\n")
            f.write("-" * 50 + "\n")
     
    def save_fold_predictions(self, fold, valid_predictions, valid_targets, fold_df, class_weights=None):
        pred_df = pd.DataFrame({
            'patient_id': fold_df['patient_id'],
            'image_id': fold_df['image_id'],
            'true_label': valid_targets,
            'prediction': valid_predictions,
            'fold': fold
        })
        if class_weights is not None:
            pred_df['class_weight_0'] = class_weights[0]
            pred_df['class_weight_1'] = class_weights[1]
        
        pred_path = self.monitor_dir / f'fold{fold}_predictions.csv'
        pred_df.to_csv(pred_path, index=False)
    
    def analyze_predictions(self, predictions_path):
        """Analyze model predictions"""
        pred_df = pd.read_csv(predictions_path)
        
        # Calculate metrics
        metrics = {
            'auc_score': roc_auc_score(pred_df['true_label'], pred_df['prediction']),
            'avg_prediction': pred_df['prediction'].mean(),
            'std_prediction': pred_df['prediction'].std(),
            'positive_rate': (pred_df['prediction'] > 0.5).mean(),
            'true_positive_rate': (
                (pred_df['prediction'] > 0.5) & 
                (pred_df['true_label'] == 1)
            ).mean()
        }
        
        return metrics
    
    def monitor_gpu_usage(self):
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
                gpu_memory.append({
                    'device': i,
                    'allocated_mb': memory_allocated,
                    'reserved_mb': memory_reserved
                })
            return gpu_memory
        return None
    
    def save_batch_metrics(self, batch_idx, metrics, fold, epoch):
        """Save batch-level metrics"""
        metrics_file = self.monitor_dir / f'fold{fold}_epoch{epoch}_batch_metrics.csv'
        
        pd.DataFrame([{
            'batch': batch_idx,
            **metrics
        }]).to_csv(metrics_file, mode='a', header=not metrics_file.exists(), index=False)


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
    """Enhanced DataLoader creation with better error handling"""
    try:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            drop_last=is_train,
            persistent_workers=False,
            timeout=60,  # Add timeout
            prefetch_factor=2 if CFG.num_workers > 0 else None,
        )
    except Exception as e:
        print(f"Error creating DataLoader: {str(e)}")
        # Fallback to most basic configuration
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False
        )

def get_predictions(model, dataset, device, batch_size=32):
    """Get predictions using batched inference"""
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for images in tqdm(dataloader, desc='Getting predictions'):
            if isinstance(images, (tuple, list)):
                images = images[0]
            images = images.to(device)
            with autocast():
                preds = model(images).sigmoid().cpu().numpy()
            predictions.append(preds)
    return np.concatenate(predictions)

def train_model():
    """Main training loop with enhanced error handling, monitoring and class balance handling"""
    # Set seeds for reproducibility
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Initialize training metrics and monitor
    fold_scores = []
    best_scores = []
    monitor = TrainingMonitor(CFG)
    
    try:
        # Read and prepare data
        train_df = pd.read_csv(CFG.base_path / 'train.csv')
        if CFG.debug:
            train_df = train_df.head(100)
            print("Debug mode: Using only 100 training samples")
        
        # Print initial class distribution and imbalance ratio
        print("\nOverall class distribution:")
        class_dist = train_df['cancer'].value_counts(normalize=True)
        print(class_dist)
        imbalance_ratio = class_dist[0] / class_dist[1]
        print(f"Imbalance ratio (negative:positive): {imbalance_ratio:.2f}:1")
        
        # Create stratified folds
        skf = StratifiedGroupKFold(
            n_splits=CFG.num_folds, 
            shuffle=True, 
            random_state=CFG.seed
        )
        
        # Create folds while maintaining patient groups
        train_df['fold'] = -1
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_df, train_df['cancer'], groups=train_df['patient_id'])
        ):
            train_df.loc[val_idx, 'fold'] = fold
        
        # Save fold assignments for reproducibility
        train_df.to_csv(CFG.processed_dir / 'fold_assignments.csv', index=False)
        
        # Training loop for each fold
        for fold in range(CFG.num_folds):
            print(f'\n{"="*20} Fold {fold + 1}/{CFG.num_folds} {"="*20}')
            
            # Clear memory before each fold
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            try:
                # Prepare fold data
                train_fold = train_df[train_df.fold != fold].reset_index(drop=True)
                valid_fold = train_df[train_df.fold == fold].reset_index(drop=True)
                
                # Print fold-specific class distributions
                print("\nTrain fold class distribution:")
                train_dist = train_fold['cancer'].value_counts(normalize=True)
                print(train_dist)
                print("\nValid fold class distribution:")
                print(valid_fold['cancer'].value_counts(normalize=True))
                
                # Create datasets with balanced sampling
                train_dataset = BalancedRSNADataset(
                    train_fold, 
                    transform=A.Compose(CFG.train_aug_list),
                    is_train=True
                )
                valid_dataset = BalancedRSNADataset(
                    valid_fold, 
                    transform=A.Compose(CFG.valid_aug_list),
                    is_train=False
                )
                
                # Create dataloaders with balanced sampling for training
                train_loader = get_balanced_dataloader(
                    train_dataset, 
                    CFG.train_batch_size, 
                    is_train=True
                )
                valid_loader = get_balanced_dataloader(
                    valid_dataset, 
                    CFG.valid_batch_size, 
                    shuffle=False, 
                    is_train=False
                )
                
                # Initialize model and move to device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = RSNAModel().to(device)
                
                # Print model summary
                monitor.print_model_summary(model)
                
                # Initialize Focal Loss with class balancing
                criterion = FocalLoss(
                    alpha=CFG.focal_loss_alpha,
                    gamma=CFG.focal_loss_gamma
                ).to(device)
                
                # Initialize optimizer
                optimizer = getattr(torch.optim, CFG.optimizer)(
                    model.parameters(),
                    lr=CFG.learning_rate,
                    weight_decay=CFG.weight_decay
                )
                
                # Initialize scheduler
                scheduler = getattr(torch.optim.lr_scheduler, CFG.scheduler)(
                    optimizer,
                    T_max=CFG.T_max,
                    eta_min=CFG.min_lr
                )
                
                # Training loop
                best_score = float('-inf')
                best_loss = float('inf')
                patience_counter = 0
                fold_history = {
                    'train_loss': [],
                    'valid_loss': [],
                    'valid_score': [],
                    'class_distribution': []  # Track class distribution
                }
                
                for epoch in range(CFG.epochs):
                    print(f'\nEpoch {epoch + 1}/{CFG.epochs}')
                    
                    # Monitor GPU usage and class distribution
                    gpu_stats = monitor.monitor_gpu_usage()
                    if gpu_stats:
                        print("\nGPU Memory Usage:")
                        for stat in gpu_stats:
                            print(f"Device {stat['device']}: "
                                  f"{stat['allocated_mb']:.0f}MB allocated, "
                                  f"{stat['reserved_mb']:.0f}MB reserved")
                    
                    try:
                        # Training phase with class balance monitoring
                        train_loss = train_one_epoch(
                            model, train_loader, criterion,
                            optimizer, scheduler, device
                        )
                        
                        # Validation phase
                        valid_loss, valid_score = valid_one_epoch(
                            model, valid_loader, criterion, device
                        )
                        
                        # Update history with class distribution
                        fold_history['train_loss'].append(train_loss)
                        fold_history['valid_loss'].append(valid_loss)
                        fold_history['valid_score'].append(valid_score)
                        fold_history['class_distribution'].append(
                            train_dataset.get_class_distribution()
                        )
                        
                        # Save batch metrics with class balance info
                        monitor.save_batch_metrics(
                            epoch, 
                            {
                                'train_loss': train_loss,
                                'valid_loss': valid_loss,
                                'valid_score': valid_score,
                                'class_ratio': train_dataset.get_class_ratio()
                            },
                            fold,
                            epoch
                        )
                        
                        # Print metrics with class balance info
                        print(
                            f'Train Loss: {train_loss:.4f} '
                            f'Valid Loss: {valid_loss:.4f} '
                            f'Valid Score: {valid_score:.4f} '
                            f'Class Ratio: {train_dataset.get_class_ratio():.2f}'
                        )
                        
                        # Save best model and visualize
                        if valid_score > best_score:
                            best_score = valid_score
                            best_loss = valid_loss
                            
                            # Save model with class balance info
                            torch.save(
                                {
                                    'epoch': epoch,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'best_score': best_score,
                                    'best_loss': best_loss,
                                    'fold_history': fold_history,
                                    'class_weights': train_dataset.class_weights.cpu().numpy()
                                },
                                CFG.model_dir / f'fold{fold}_best.pth'
                            )
                            print(f'Best model saved! Score: {best_score:.4f}')
                            
                            # Visualization and logging
                            fig = monitor.visualize_training_progress(fold_history)
                            fig.savefig(monitor.monitor_dir / f'fold{fold}_training_progress.png')
                            plt.close(fig)
                            
                            monitor.log_fold_metrics(fold, fold_history, best_score, best_loss)
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            
                        # Early stopping check
                        if patience_counter >= CFG.patience:
                            print(f'Early stopping triggered after {epoch + 1} epochs')
                            break
                            
                    except Exception as e:
                        print(f"Error in epoch {epoch + 1}: {str(e)}")
                        print(traceback.format_exc())
                        break
                
                # Store fold results and save predictions
                fold_scores.append(best_score)
                best_scores.append({
                    'fold': fold,
                    'score': best_score,
                    'loss': best_loss,
                    'final_class_ratio': train_dataset.get_class_ratio()
                })
                
                # Save validation predictions with class balance metrics
                valid_preds = get_predictions(model, valid_dataset, device, CFG.valid_batch_size)
                valid_labels = valid_dataset.get_all_labels()
                monitor.save_fold_predictions(
                    fold, 
                    valid_preds, 
                    valid_labels, 
                    valid_fold, 
                    class_weights=train_dataset.class_weights.cpu().numpy()
                )
                
            except Exception as e:
                print(f"Error in fold {fold + 1}: {str(e)}")
                print(traceback.format_exc())
                continue
                
            finally:
                # Cleanup
                try:
                    del model, train_loader, valid_loader
                    del train_dataset, valid_dataset
                    del optimizer, scheduler
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error in cleanup: {str(e)}")
        
        # Print final results with class balance metrics
        print("\nTraining completed!")
        print("\nBest scores per fold:")
        for score_dict in best_scores:
            print(
                f"Fold {score_dict['fold'] + 1}: "
                f"Score = {score_dict['score']:.4f}, "
                f"Loss = {score_dict['loss']:.4f}, "
                f"Class Ratio = {score_dict['final_class_ratio']:.2f}"
            )
        
        print(f"\nMean CV score: {np.mean(fold_scores):.4f}")
        print(f"Std CV score: {np.std(fold_scores):.4f}")
        
        # Analyze overall predictions with class balance consideration
        for fold in range(CFG.num_folds):
            pred_path = monitor.monitor_dir / f'fold{fold}_predictions.csv'
            if pred_path.exists():
                metrics = monitor.analyze_predictions(pred_path)
                print(f"\nFold {fold + 1} Prediction Analysis:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")
        
        return best_scores
        
    except Exception as e:
        print(f"Fatal error in training: {str(e)}")
        print(traceback.format_exc())
        return None

def inference():
    """Performs inference using trained models"""
    print("\nStarting inference...")
    
    try:
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
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True
        )
        
        # Inference with all folds
        for fold in range(CFG.num_folds):
            print(f'Inferencing fold {fold + 1}/{CFG.num_folds}')
            model = RSNAModel().to(device)
            
            try:
                # Load the saved model state
                checkpoint = torch.load(
                    CFG.model_dir / f'fold{fold}_best.pth',
                    map_location=device
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                fold_preds = []
                with torch.no_grad():
                    for images in tqdm(test_loader, desc=f'Fold {fold + 1}'):
                        images = images.to(device)
                        with autocast():  # Add mixed precision inference
                            y_preds = model(images).squeeze(1)
                        fold_preds.append(y_preds.sigmoid().cpu().numpy())
                
                fold_preds = np.concatenate(fold_preds)
                predictions.append(fold_preds)
                print(f"Fold {fold + 1} inference completed")
                
            except Exception as e:
                print(f"Error in fold {fold + 1} inference: {str(e)}")
                print(traceback.format_exc())
                continue
            
            finally:
                # Cleanup
                try:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error in cleanup: {str(e)}")
        
        if not predictions:
            raise ValueError("No valid predictions from any fold")
            
        # Average predictions from all folds
        predictions = np.mean(predictions, axis=0)
        
        # Create submission
        submission = pd.DataFrame({
            'prediction_id': test_df['prediction_id'],
            'cancer': predictions
        })
        
        # Save submission with timestamp
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        submission_path = f'submission_{timestamp}.csv'
        submission.to_csv(submission_path, index=False)
        print(f'Submission saved to {submission_path}!')
        
        return submission
        
    except Exception as e:
        print(f"Fatal error in inference: {str(e)}")
        print(traceback.format_exc())
        return None

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

def get_balanced_dataloader(dataset, batch_size, shuffle=True, is_train=True):
    """Creates DataLoader with balanced sampling if needed"""
    if is_train:
        sampler = dataset.get_sampler()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=CFG.num_workers,
            pin_memory=CFG.pin_memory,
            drop_last=is_train,
            persistent_workers=CFG.persistent_workers
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=CFG.num_workers,
            pin_memory=CFG.pin_memory,
            drop_last=is_train,
            persistent_workers=CFG.persistent_workers
        )

def save_run_config(cfg, run_dir):
    """Save training configuration and run info"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = run_dir / f'run_config_{timestamp}.txt'
    
    with open(config_path, 'w') as f:
        f.write(f"Run started at: {timestamp}\n\n")
        f.write("Training Configuration:\n")
        f.write("-" * 50 + "\n")
        
        # Save all CFG attributes
        for attr_name in dir(cfg):
            if not attr_name.startswith('__'):
                attr_value = getattr(cfg, attr_name)
                if isinstance(attr_value, Path):
                    attr_value = str(attr_value)
                f.write(f"{attr_name}: {attr_value}\n")
        
        # Save system info
        f.write("\nSystem Information:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA version: {torch.version.cuda}\n")
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            
        return config_path

def main():
    """Main execution function with enhanced logging and error handling"""
    start_time = datetime.now()
    print(f"Starting RSNA Mammography Pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Create run directory with timestamp
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        run_dir = CFG.model_dir / f'run_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save run configuration
        config_path = save_run_config(CFG, run_dir)
        print(f"Run configuration saved to: {config_path}")
        
        # Create necessary directories
        CFG.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        log_path = run_dir / 'run_log.txt'
        logging.basicConfig(
            filename=str(log_path),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Pipeline started")
        
        try:
            # Step 1: Process training data
            print("\nStep 1: Processing training data...")
            logging.info("Starting training data processing")
            
            train_df = pd.read_csv(CFG.base_path / 'train.csv')
            if CFG.debug:
                train_df = train_df.head(100)
                print("Debug mode: Using only 100 training samples")
                logging.info("Running in debug mode with 100 samples")
            
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
            
            processing_msg = f"Training data processing completed. Processed: {processed_count}, Failed: {failed_count}"
            print(processing_msg)
            logging.info(processing_msg)
            
            # Step 2: Train models
            print("\nStep 2: Training models...")
            logging.info("Starting model training")
            best_scores = train_model()
            
            if best_scores:
                mean_cv = np.mean([score['score'] for score in best_scores])
                logging.info(f"Training completed. Mean CV score: {mean_cv:.4f}")
            
            # Step 3: Process test data
            print("\nStep 3: Processing test data...")
            logging.info("Starting test data processing")
            process_test_data()
            
            # Step 4: Generate predictions
            print("\nStep 4: Generating predictions...")
            logging.info("Starting inference")
            submission = inference()
            
            if submission is not None:
                submission_stats = {
                    'mean': submission['cancer'].mean(),
                    'std': submission['cancer'].std(),
                    'min': submission['cancer'].min(),
                    'max': submission['cancer'].max()
                }
                logging.info(f"Submission statistics: {submission_stats}")
            
            # Calculate and log total runtime
            end_time = datetime.now()
            runtime = end_time - start_time
            runtime_msg = f"\nPipeline completed successfully! Total runtime: {runtime}"
            print(runtime_msg)
            logging.info(runtime_msg)
            
            if CFG.debug:
                debug_msg = "\nNote: This was run in debug mode. Set CFG.debug = False for full training."
                print(debug_msg)
                logging.info(debug_msg)
            
            # Save final summary
            with open(run_dir / 'run_summary.txt', 'w') as f:
                f.write(f"Run Summary\n")
                f.write(f"===========\n")
                f.write(f"Start time: {start_time}\n")
                f.write(f"End time: {end_time}\n")
                f.write(f"Total runtime: {runtime}\n")
                f.write(f"Processed images: {processed_count}\n")
                f.write(f"Failed images: {failed_count}\n")
                if best_scores:
                    f.write(f"Mean CV score: {mean_cv:.4f}\n")
                if submission is not None:
                    f.write(f"\nSubmission Statistics:\n")
                    for stat, value in submission_stats.items():
                        f.write(f"{stat}: {value:.4f}\n")
            
            return submission
            
        except Exception as e:
            error_msg = f"Pipeline step error: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            return None
            
    except Exception as e:
        error_msg = f"Fatal error in pipeline initialization: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Ensure proper multiprocessing behavior
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()