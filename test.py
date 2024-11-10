
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
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['cancer'], groups=train_df['patient_id'])):
        train_df.loc[val_idx, 'fold'] = fold
    
    # Training loop
    for fold in range(CFG.num_folds):
        print(f'\nTraining fold {fold + 1}/{CFG.num_folds}')
        
        # Check class distribution in fold
        train_fold = train_df[train_df.fold != fold]
        valid_fold = train_df[train_df.fold == fold]
        
        print("\nTrain fold class distribution:")
        print(train_fold['cancer'].value_counts(normalize=True))
        print("\nValid fold class distribution:")
        print(valid_fold['cancer'].value_counts(normalize=True))
        
        # Prepare data
        train_loader = DataLoader(
            RSNADataset(
                train_fold,
                transform=A.Compose(CFG.train_aug_list)
            ),
            batch_size=CFG.train_batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True
        )
        
        valid_loader = DataLoader(
            RSNADataset(
                valid_fold,
                transform=A.Compose(CFG.valid_aug_list)
            ),
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True
        )
        
