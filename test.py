class CFG:
    # ... other configs ...
    
    # Preprocessing
    image_size = 512  # Single value for square images
    target_size = (512, 512)  # For image resizing
    output_format = 'png'
    
    # Model configs
    model_name = 'efficientnet_b3'
    pretrained = True
    num_classes = 1  # Add this and remove the old target_size = 1