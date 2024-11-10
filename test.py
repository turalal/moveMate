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