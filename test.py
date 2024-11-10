def read_dicom(self, patient_id, image_id, is_train=True):
    try:
        images_path = self.train_images_path if is_train else self.test_images_path
        dicom_path = images_path / str(patient_id) / f"{image_id}.dcm"
        
        if not dicom_path.exists():
            print(f"File not found: {dicom_path}")
            return None
            
        # Simple DICOM reading without GDCM check
        dicom = pydicom.dcmread(str(dicom_path), force=True)
        
        try:
            # Try to access pixel_array
            img = dicom.pixel_array
        except Exception as e:
            print(f"Error reading pixel array: {str(e)}")
            return None
            
        # Convert to float and normalize
        img = img.astype(float)
        
        # Normalize
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