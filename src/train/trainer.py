from src.utils.logger import logger

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.models.yolo_detection_model import YOLODetectionModel
from src.dataloaders.yolo_dataloader import YOLODataModule

class TrainerWrapper:
    def __init__(self, config):
        self.logger = logger
        self.logger.info("Trainer initialized with model and data module.")
        self.config = config
        self.model = YOLODetectionModel(config)
        self.data_module = YOLODataModule(config)

    def setup_callbacks(self):
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
        self.logger.info("Setting up PyTorch Lightning Trainer...")

        # Determine if GPU is available
        if torch.cuda.is_available() and self.config['training'].get('gpus', 1) > 0:
            accelerator = 'gpu'
            devices = self.config['training'].get('gpus', 1)
        else:
            accelerator = 'cpu'
            devices = 'auto'  # Fallback to CPU

        # Set up callbacks
        callbacks = self.setup_callbacks()

        # Create PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=self.config['training']['epochs'],
            devices=devices,
            accelerator=accelerator,
            precision=16 if self.config['training'].get('use_mixed_precision', False) else 32,
            callbacks=callbacks,
            default_root_dir=self.config['output']['output_dir']
        )
        return trainer

    def train(self):
        self.logger.info("Starting training...")

        trainer = self.setup_trainer()
        # Start training
        trainer.fit(self.model, datamodule=self.data_module)
        # Return validation loss for HPO
        return trainer.callback_metrics['val_loss'].item()
