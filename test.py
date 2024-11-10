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