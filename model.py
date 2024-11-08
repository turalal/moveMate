import numpy as np
import pandas as pd
import pydicom
import cv2
from pathlib import Path
from sklearn.model_selection import GroupKFold
import albumentations as A
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os

class RSNAPreprocessor:
    def __init__(self, base_path="/kaggle/input/rsna-breast-cancer-detection", 
                 target_size=(2048, 2048),
                 output_format='png'):
        self.base_path = Path(base_path)
        self.train_images_path = self.base_path / "train_images"
        self.test_images_path = self.base_path / "test_images"
        self.target_size = target_size
        self.output_format = output_format.lower()
        
        if self.output_format not in ['png', 'jpg', 'jpeg']:
            raise ValueError("output_format must be 'png' or 'jpg'/'jpeg'")

    def get_dicom_path(self, patient_id, image_id, is_train=True):
        """
        Get the path to a DICOM file
        """
        images_path = self.train_images_path if is_train else self.test_images_path
        return images_path / str(patient_id) / f"{image_id}"

    def read_dicom(self, patient_id, image_id, is_train=True):
        """
        Read and preprocess DICOM image
        """
        dicom_path = self.get_dicom_path(patient_id, image_id, is_train)
        try:
            # Add .dcm extension if not in the image_id
            if not str(dicom_path).endswith('.dcm'):
                dicom_path = Path(str(dicom_path) + '.dcm')

            print(f"Reading DICOM from: {dicom_path}")  # Debug print
            
            if not dicom_path.exists():
                print(f"File not found: {dicom_path}")
                return None
                
            dicom = pydicom.dcmread(dicom_path)
            
            # Process image
            img = dicom.pixel_array
            
            # Convert to float and normalize
            img = img.astype(float)
            if img.max() != img.min():
                img = (img - img.min()) / (img.max() - img.min())
            
            # Scale to 0-255 range
            img = (img * 255).astype(np.uint8)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            # Resize while maintaining aspect ratio
            aspect = img.shape[0] / img.shape[1]
            if aspect > 1:
                new_height = self.target_size[0]
                new_width = int(new_height / aspect)
            else:
                new_width = self.target_size[1]
                new_height = int(new_width * aspect)
            
            img = cv2.resize(img, (new_width, new_height))
            
            # Add padding to reach target size
            top_pad = (self.target_size[0] - img.shape[0]) // 2
            bottom_pad = self.target_size[0] - img.shape[0] - top_pad
            left_pad = (self.target_size[1] - img.shape[1]) // 2
            right_pad = self.target_size[1] - img.shape[1] - left_pad
            
            img = cv2.copyMakeBorder(
                img, top_pad, bottom_pad, left_pad, right_pad,
                cv2.BORDER_CONSTANT, value=0
            )
            
            return img

        except Exception as e:
            print(f"Error processing image {image_id} for patient {patient_id}: {str(e)}")
            return None

    def save_image(self, img, output_path):
        """
        Save processed image in specified format
        """
        if img is not None:
            if self.output_format == 'png':
                cv2.imwrite(str(output_path.with_suffix('.png')), img)
            else:  # jpg/jpeg
                cv2.imwrite(str(output_path.with_suffix('.jpg')), img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    def process_and_save(self, metadata_df, output_dir, num_samples=None):
        """
        Process images and save them in the specified format
        """
        if num_samples:
            metadata_df = metadata_df.head(num_samples)
        
        # Create output directory structure
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different views
        for view in ['CC', 'MLO']:
            (output_dir / view).mkdir(exist_ok=True)
            (output_dir / view / 'L').mkdir(exist_ok=True)
            (output_dir / view / 'R').mkdir(exist_ok=True)
        
        processed_count = 0
        failed_count = 0
        
        print("\nProcessing metadata shape:", metadata_df.shape)
        print("Sample row:")
        print(metadata_df.iloc[0])
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            try:
                img = self.read_dicom(
                    patient_id=str(row['patient_id']),  # Convert to string
                    image_id=str(row['image_id'])       # Convert to string
                )
                
                if img is not None:
                    # Create organized directory structure based on view and laterality
                    view = row['view']      # CC or MLO
                    laterality = row['laterality']  # L or R
                    
                    # Define output path with organized structure
                    output_path = output_dir / view / laterality / f"{row['patient_id']}_{row['image_id']}"
                    
                    # Save the image
                    self.save_image(img, output_path)
                    processed_count += 1
                    
                    # Save a thumbnail for quick viewing
                    thumbnail = cv2.resize(img, (512, 512))
                    thumbnail_path = output_path.with_name(f"{output_path.stem}_thumb")
                    self.save_image(thumbnail, thumbnail_path)
                    
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                print(f"Error processing row {idx}: {str(e)}")
                continue
                
        return processed_count, failed_count

def main():
    print("Initializing RSNA Mammography Preprocessing...")
    
    # Initialize preprocessor with PNG output format
    preprocessor = RSNAPreprocessor(output_format='png')
    
    try:
        # Read metadata
        train_df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")
        print(f"Total images to process: {len(train_df)}")
        
        # Create output directory
        output_dir = Path("/kaggle/working/processed_images")
        
        # Process images
        print("\nProcessing images...")
        processed_count, failed_count = preprocessor.process_and_save(
            train_df,
            output_dir,
            num_samples=5  # Change this number or set to None for all images
        )
        
        print(f"\nProcessing completed:")
        print(f"Successfully processed: {processed_count}")
        print(f"Failed: {failed_count}")
        
        # Save processing summary
        summary = {
            'total_images': len(train_df),
            'processed': processed_count,
            'failed': failed_count,
            'success_rate': processed_count / (processed_count + failed_count) * 100 if (processed_count + failed_count) > 0 else 0
        }
        
        pd.DataFrame([summary]).to_csv(output_dir / 'processing_summary.csv', index=False)
        print("\nProcessing summary saved.")
        
        print("\nOutput directory structure:")
        print(f"{output_dir}/")
        print("├── CC/")
        print("│   ├── L/")
        print("│   └── R/")
        print("├── MLO/")
        print("│   ├── L/")
        print("│   └── R/")
        print("└── processing_summary.csv")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()