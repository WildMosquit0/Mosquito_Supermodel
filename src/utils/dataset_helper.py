import os

def get_dataset_paths(config):
    """
    Helper function to validate and return dataset paths for training and validation.

    Args:
        config (dict): Configuration dictionary that includes the dataset paths for training and validation.

    Returns:
        tuple: A tuple containing (train_input, val_input), which are the validated paths to the training and validation datasets.
    """
    # Extract dataset paths from config
    train_input = config['input']['train']
    val_input = config['input']['val']

    # Check if training input is valid
    if not os.path.exists(train_input):
        raise FileNotFoundError(f"Training dataset not found at {train_input}")
    if os.path.isdir(train_input):
        print(f"Training with dataset from directory: {train_input}")
    elif os.path.isfile(train_input) and train_input.endswith('.txt'):
        print(f"Training with dataset from file list: {train_input}")
    else:
        raise ValueError(f"Invalid format for training dataset: {train_input}. Must be a directory or a .txt file.")

    # Check if validation input is valid
    if not os.path.exists(val_input):
        raise FileNotFoundError(f"Validation dataset not found at {val_input}")
    if os.path.isdir(val_input):
        print(f"Validating with dataset from directory: {val_input}")
    elif os.path.isfile(val_input) and val_input.endswith('.txt'):
        print(f"Validating with dataset from file list: {val_input}")
    else:
        raise ValueError(f"Invalid format for validation dataset: {val_input}. Must be a directory or a .txt file.")

    return train_input, val_input
