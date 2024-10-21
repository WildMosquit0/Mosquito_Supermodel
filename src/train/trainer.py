from src.utils.logger import logger
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.models.yolo_detection_model import YOLODetectionModel
from src.dataloaders.yolo_dataloader import YOLODataModule

class TrainerWrapper:
    """
    TrainerWrapper orchestrates training, validation, and checkpointing for the model using PyTorch Lightning.
    """
    
    def __init__(self, config: dict):
        self.logger = logger
        self.config = config
        self.logger.info("Initializing model and data module...")
        self.model = YOLODetectionModel(config)
        self.data_module = YOLODataModule(config)

    def setup_callbacks(self) -> list:
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

    def setup_trainer(self) -> pl.Trainer:
        accelerator = 'gpu' if torch.cuda.is_available() and self.config['training'].get('gpus', 1) > 0 else 'cpu'
        devices = self.config['training'].get('gpus', 1) if accelerator == 'gpu' else 'auto'
        callbacks = self.setup_callbacks()
        output_dir = self.config['output'].get('output_dir', './output')
        trainer = pl.Trainer(
            max_epochs=self.config['training'].get('epochs', 10),
            devices=devices,
            accelerator=accelerator,
            precision=16 if self.config['training'].get('use_mixed_precision', False) else 32,
            callbacks=callbacks,
            default_root_dir=output_dir
        )
        return trainer

    def train(self) -> None:
        self.logger.info("Starting training...")
        trainer = self.setup_trainer()
        trainer.fit(self.model, datamodule=self.data_module)
