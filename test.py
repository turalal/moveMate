class RSNAModel(nn.Module):
    """Model architecture"""
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            CFG.model_name, 
            pretrained=CFG.pretrained, 
            num_classes=CFG.num_classes  # Changed from target_size
        )
        
    def forward(self, x):
        return self.model(x)