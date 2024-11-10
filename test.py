def get_dataloader(dataset, batch_size, shuffle=True, is_train=True):
    """Safe DataLoader creation with proper worker initialization"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=False,  # Disable persistent workers
    )

class CFG:
    """Configuration class containing all parameters"""
    # Previous settings remain...
    
    # Modified training parameters
    num_workers = 0  # Changed from 4 to 0
    train_batch_size = 8  # Reduced from 16
    valid_batch_size = 16  # Reduced from 32
    
    # Add safety flags
    persistent_workers = False
    pin_memory = True
    
def train_model():
    """Main training loop with enhanced error handling"""
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    
    # Force single-process data loading
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    try:
        # Read and prepare data
        train_df = pd.read_csv(CFG.base_path / 'train.csv')
        if CFG.debug:
            train_df = train_df.head(100)
        
        # Create stratified folds
        skf = StratifiedGroupKFold(
            n_splits=CFG.num_folds, 
            shuffle=True, 
            random_state=CFG.seed
        )
        
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_df, train_df['cancer'], 
                     groups=train_df['patient_id'])):
            
            print(f'\nTraining fold {fold + 1}/{CFG.num_folds}')
            
            # Clear GPU memory before each fold
            torch.cuda.empty_cache()
            gc.collect()
            
            try:
                # Prepare datasets with proper error handling
                train_dataset = RSNADataset(
                    train_df.iloc[train_idx].reset_index(drop=True),
                    transform=A.Compose(CFG.train_aug_list)
                )
                valid_dataset = RSNADataset(
                    train_df.iloc[val_idx].reset_index(drop=True),
                    transform=A.Compose(CFG.valid_aug_list)
                )
                
                # Create dataloaders with safe settings
                train_loader = get_dataloader(
                    train_dataset, 
                    CFG.train_batch_size, 
                    shuffle=True
                )
                valid_loader = get_dataloader(
                    valid_dataset, 
                    CFG.valid_batch_size, 
                    shuffle=False, 
                    is_train=False
                )
                
                # Training loop implementation
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = RSNAModel().to(device)
                
                # Rest of the training code remains the same...
                
            except Exception as e:
                print(f"Error in fold {fold + 1}: {str(e)}")
                continue
                
            finally:
                # Proper cleanup
                try:
                    del model
                    del train_loader
                    del valid_loader
                    del train_dataset
                    del valid_dataset
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass
                
    except Exception as e:
        print(f"Error in training: {str(e)}")
        return None

if __name__ == "__main__":
    # Ensure proper multiprocessing behavior
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()