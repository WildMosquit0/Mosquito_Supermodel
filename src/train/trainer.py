from src.utils.logger import logger
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.models.yolo_detection_model import YOLODetectionModel
from src.dataloaders.yolo_dataloader import YOLODataModule

class TrainerWrapper:
    def __init__(self, config):
        self.logger = logger
        self.config = config

        # Initialize the model and data module
        self.logger.info("Initializing model and data module...")
        self.model = YOLODetectionModel(config)  # Use YOLO model via PyTorch Lightning
        self.data_module = YOLODataModule(config)

    def setup_callbacks(self):
        """Set up checkpoint and early stopping callbacks."""
        self.logger.info("Setting up callbacks...")

        checkpoint_callback = ModelCheckpoint(
            monitor=self.config['output'].get('monitor_metric', 'val_loss'),
            dirpath=self.config['output']['output_dir'],
            filename=self.config['output'].get('checkpoint_filename', 'best-checkpoint'),
            save_top_k=self.config['output'].get('save_top_k', 1),
            mode=self.config['output'].get('checkpoint_mode', 'min')
        )
        early_stopping_callback = EarlyStopping(
            monitor=self.config['output'].get('monitor_metric', 'val_loss'),
            patience=self.config['training'].get('early_stopping_patience', 3),
            mode=self.config['output'].get('checkpoint_mode', 'min')
        )
        return [checkpoint_callback, early_stopping_callback]

    def setup_trainer(self):
        """Set up the PyTorch Lightning Trainer."""
        self.logger.info("Setting up PyTorch Lightning Trainer...")

        # Check for GPU availability
        accelerator = 'gpu' if torch.cuda.is_available() and self.config['training'].get('gpus', 1) > 0 else 'cpu'
        devices = self.config['training'].get('gpus', 1) if accelerator == 'gpu' else 'auto'

        # Set up callbacks
        callbacks = self.setup_callbacks()

        # Create the PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=self.config['training'].get('epochs', 10),  # Set epochs from config
            devices=devices,
            accelerator=accelerator,
            precision=16 if self.config['training'].get('use_mixed_precision', False) else 32,
            callbacks=callbacks,
            default_root_dir=self.config['output']['output_dir']
        )
        return trainer

    def train(self):
        """Start the training process."""
        self.logger.info("Starting training...")

        # Set up the trainer
        trainer = self.setup_trainer()

        # Start training the model
        trainer.fit(self.model, datamodule=self.data_module)

        # Return validation loss if it exists (useful for hyperparameter optimization)
        if 'val_loss' in trainer.callback_metrics:
            return trainer.callback_metrics['val_loss'].item()
        else:
            self.logger.warning("Validation loss not found in callback metrics.")
            return None
