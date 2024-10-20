import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.dataloaders.yolo_dataset import YOLODataset
from src.utils.transforms import get_transforms

class YOLODataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.batch_size = config['training'].get('batch_size', 16)

        # Handle input sources from the config
        self.train_input = config['input']['train']
        self.val_input = config['input']['val']

        # Generate transformations
        self.transform = get_transforms(config.get('transforms', None))

    def setup(self, stage=None):
        """
        Setup datasets for training and validation. Called by Lightning before training/validation.
        """
        # Train dataset
        self.train_dataset = YOLODataset(input_path=self.train_input, transform=self.transform)

        # Validation dataset
        self.val_dataset = YOLODataset(input_path=self.val_input, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
