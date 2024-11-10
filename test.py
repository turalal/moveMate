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

class RSNAPreprocessor:
    """Enhanced preprocessing with parallel processing and better error handling"""
    def __init__(self, **kwargs):
        # Previous initialization code remains...
        self.error_log = []  # Add error logging
        
    def process_and_save(self, metadata_df, output_dir, num_samples=None):
        if num_samples:
            metadata_df = metadata_df.head(num_samples)
        
        output_dir = Path(output_dir)
        self._create_directory_structure(output_dir)
        
        processed_count = 0
        failed_count = 0
        
        # Process in smaller batches to manage memory
        batch_size = 50
        num_batches = (len(metadata_df) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(metadata_df))
            batch_df = metadata_df.iloc[start_idx:end_idx]
            
            # Process batch
            for idx, row in tqdm(batch_df.iterrows(), 
                               total=len(batch_df),
                               desc=f'Processing batch {batch_idx + 1}/{num_batches}'):
                try:
                    img = self.read_dicom(
                        patient_id=str(row['patient_id']),
                        image_id=str(row['image_id'])
                    )
                    
                    if img is not None and img.size > 0:
                        output_path = (output_dir / row['view'] / row['laterality'] / 
                                     f"{row['patient_id']}_{row['image_id']}")
                        success = self.save_image(img, output_path)
                        
                        if success:
                            thumbnail = cv2.resize(img, (512, 512))
                            thumbnail_path = output_path.with_name(f"{output_path.stem}_thumb")
                            self.save_image(thumbnail, thumbnail_path)
                            processed_count += 1
                        else:
                            failed_count += 1
                            self.error_log.append({
                                'patient_id': row['patient_id'],
                                'image_id': row['image_id'],
                                'error': 'Failed to save image'
                            })
                    else:
                        failed_count += 1
                        self.error_log.append({
                            'patient_id': row['patient_id'],
                            'image_id': row['image_id'],
                            'error': 'Invalid image data'
                        })
                        
                except Exception as e:
                    failed_count += 1
                    self.error_log.append({
                        'patient_id': row['patient_id'],
                        'image_id': row['image_id'],
                        'error': str(e)
                    })
                    print(f"Error processing row {idx}: {str(e)}")
            
            # Clear memory after each batch
            gc.collect()
        
        # Save error log
        if self.error_log:
            error_df = pd.DataFrame(self.error_log)
            error_df.to_csv(output_dir / 'processing_errors.csv', index=False)
            print(f"\nError log saved with {len(self.error_log)} entries")
        
        return processed_count, failed_count

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