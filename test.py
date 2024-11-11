# Add these imports at the top
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

class BalancedRSNADataset(Dataset):
    """Enhanced Dataset with class balancing capabilities"""
    def __init__(self, df, transform=None, is_train=True):
        self.df = df
        self.transform = transform
        self.is_train = is_train
        self.image_cache = {}
        
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

def train_model():
    """Updated training function with balanced sampling and focal loss"""
    # Previous setup code remains the same...
    
    for fold in range(CFG.num_folds):
        try:
            # Prepare fold data
            train_fold = train_df[train_df.fold != fold].reset_index(drop=True)
            valid_fold = train_df[train_df.fold == fold].reset_index(drop=True)
            
            # Create datasets using balanced dataset class
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
            
            # Create dataloaders with balanced sampling
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
            
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = RSNAModel().to(device)
            
            # Use Focal Loss instead of BCE
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
            
            # Rest of the training loop remains the same...

# Update CFG class with additional parameters
class CFG:
    # ... (previous parameters remain the same)
    
    # Class balancing parameters
    focal_loss_alpha = 0.25
    focal_loss_gamma = 2.0
    use_class_weights = True
    oversample_minority = True