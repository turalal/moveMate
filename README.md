# RSNA Mammography Classification Pipeline

## Overview
This codebase implements an end-to-end machine learning pipeline for breast cancer detection from mammography images. The pipeline includes data preprocessing, model training with class balancing, and inference components designed to handle the RSNA Mammography Classification challenge.

## Key Features
- Robust DICOM image preprocessing with error handling
- Class-balanced training to handle data imbalance
- Mixed precision training for improved performance
- Multi-fold cross-validation
- Comprehensive monitoring and logging
- Enhanced error handling and recovery
- Memory-efficient data loading
- Advanced augmentation strategies

## Components

### 1. Configuration (CFG class)
- Controls all hyperparameters and settings
- Includes paths, model parameters, training settings
- Configurable debug mode for rapid prototyping

### 2. Data Processing (RSNAPreprocessor)
- Handles DICOM image preprocessing
- Implements robust error handling
- Features:
  - CLAHE contrast enhancement
  - Aspect ratio-preserving resizing
  - Multi-format image handling
  - Caching mechanism
  - Detailed error logging

### 3. Dataset Handling (BalancedRSNADataset, RSNADataset)
- Implements PyTorch Dataset interface
- Features:
  - Class-balanced sampling
  - Image caching
  - Augmentation pipeline
  - Error recovery
  - Memory-efficient data loading

### 4. Model Architecture (RSNAModel)
- Based on EfficientNet backbone
- Features:
  - Attention mechanism
  - Custom classification head
  - Feature extraction capabilities
  - Metric tracking
  - Model analysis tools

### 5. Training Components
- FocalLoss for handling class imbalance
- Mixed precision training
- Advanced monitoring:
  - Loss tracking
  - Metric computation
  - GPU usage monitoring
  - Class distribution tracking

### 6. Monitoring (TrainingMonitor)
- Comprehensive training visualization
- Metrics tracking
- Model performance analysis
- Resource usage monitoring

## Training Pipeline

### Setup
```python
# Required imports are handled in the code
# Configuration is managed through CFG class
```

### Data Preparation
1. DICOM images are preprocessed
2. Images are resized and enhanced
3. Train/validation splits are created using StratifiedGroupKFold

### Training Process
1. Model initialization
2. K-fold cross-validation
3. For each fold:
   - Data loading with class balancing
   - Model training with early stopping
   - Model validation
   - Best model saving
   - Performance monitoring

### Inference
1. Test data preprocessing
2. Model ensemble predictions
3. Submission file generation

## Key Parameters

### Model Configuration
- Model: EfficientNet-B3
- Image size: 512x512
- Batch sizes: 
  - Training: 8
  - Validation: 32

### Training Parameters
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-6
- Scheduler: CosineAnnealingLR
- Early stopping patience: 3

### Data Augmentation
- Random rotations
- Flips (horizontal/vertical)
- Shift/Scale/Rotate
- Noise/Blur
- Grid distortion
- Dropout

## Performance Optimization

### Memory Management
- Image caching system
- Garbage collection
- GPU memory monitoring
- Batch size optimization

### Class Imbalance Handling
- Focal Loss
- Class weights
- Balanced sampling
- Distribution monitoring

### Error Handling
- Comprehensive error logging
- Recovery mechanisms
- Detailed monitoring
- Safe fallbacks

## Usage

### Basic Usage
```python
# Run complete pipeline
python rsna_pipeline.py

# Debug mode (subset of data)
# Set CFG.debug = True
```

### Configuration
- Modify CFG class parameters
- Adjust paths for data location
- Set debug mode as needed
- Configure model and training parameters

### Output
- Trained models
- Performance metrics
- Training logs
- Visualization plots
- Submission file

## Dependencies
- PyTorch
- OpenCV
- albumentations
- pydicom
- scikit-learn
- timm
- pandas
- numpy

## Notes
- Designed for RSNA Mammography Classification
- Handles class imbalance
- Implements robust error recovery
- Memory-efficient processing
- Comprehensive monitoring

## Best Practices
1. Start with debug mode
2. Monitor GPU memory usage
3. Adjust batch sizes as needed
4. Check class balance metrics
5. Monitor validation scores
6. Review error logs regularly

## Error Handling
- Comprehensive error logging
- Graceful failure recovery
- Detailed error messages
- Safe state management

## Performance Monitoring
- Training metrics
- GPU usage
- Memory consumption
- Class distribution
- Model performance

## Contributing
- Follow existing code structure
- Maintain error handling
- Update documentation
- Add comprehensive logging
- Test thoroughly
