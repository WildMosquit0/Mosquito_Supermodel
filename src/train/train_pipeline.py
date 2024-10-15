import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.models.yolo_detection_model import YOLODetectionModel
from src.dataloaders.yolo_dataloader import YOLODataModule
from src.utils.config import load_config

def train():
    # Load the configuration from the config file
    config = load_config("config.json")
    
    # Initialize the YOLO model using the configuration
    model = YOLODetectionModel(config)
    
    # Initialize the DataLoader (YOLODataModule) with the configuration
    data_module = YOLODataModule(config)
    
    # Set up model checkpointing based on config parameters
    checkpoint_callback = ModelCheckpoint(
        monitor=config['output'].get('monitor_metric', 'val_loss'),  # Metric to monitor from config
        dirpath=config['output']['output_dir'],  # Output directory from config
        filename=config['output'].get('checkpoint_filename', 'best-checkpoint'),  # Checkpoint filename from config
        save_top_k=config['output'].get('save_top_k', 1),  # Save only the best 'k' checkpoints
        mode=config['output'].get('checkpoint_mode', 'min')  # Mode, e.g., 'min' or 'max'
    )
    
    # Set up early stopping based on the config
    early_stopping_callback = EarlyStopping(
        monitor=config['output'].get('monitor_metric', 'val_loss'),  # Metric to monitor from config
        patience=config['training'].get('early_stopping_patience', 3),  # Patience from config
        mode=config['output'].get('checkpoint_mode', 'min')  # Mode from config
    )
    
    # Create a PyTorch Lightning Trainer using config parameters
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],  # Number of epochs from config
        devices=config['training'].get('gpus', 1),  # Number of GPUs to use, or 'auto'
        accelerator="gpu" if config['training'].get('gpus', 1) > 0 else "cpu",  # Use GPU if available
        precision=16 if config['training'].get('use_mixed_precision', False) else 32,  # Mixed precision support from config
        callbacks=[checkpoint_callback, early_stopping_callback],  # Callbacks
        default_root_dir=config['output']['output_dir']  # Default output directory from config
    )
    
    # Train the model using the data module and the Trainer
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    train()
