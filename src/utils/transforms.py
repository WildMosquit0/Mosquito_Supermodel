from torchvision import transforms

def get_transforms(config=None):
    """
    Generate transformations based on the provided config. If no config is provided,
    default transformations will be applied.

    Args:
        config (dict, optional): Configuration dictionary containing transformation details.

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    transform_list = []

    if config:
        # Resize if specified in config
        if 'resize' in config:
            size = config['resize'].get('size', (640, 640))
            transform_list.append(transforms.Resize(size))

        # Normalize if specified in config
        if 'normalize' in config:
            mean = config['normalize'].get('mean', [0.571, 0.590, 0.551])  # Default mean (our benchmark)
            std = config['normalize'].get('std', [0.232, 0.239, 0.248])    # Default std (our benchmark)
            transform_list.append(transforms.Normalize(mean=mean, std=std))

    # Apply default resize and normalization if no transforms are specified
    if not transform_list:
        # Default resize and ToTensor
        transform_list.append(transforms.Resize((640, 640)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.571, 0.590, 0.551], std=[0.232, 0.239, 0.248]))
    else:
        # Ensure ToTensor is always applied
        transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)
