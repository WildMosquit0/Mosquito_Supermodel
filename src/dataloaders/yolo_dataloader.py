import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

class YOLODataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.batch_size = config['training'].get('batch_size', 16)
        self.images_dir = config['input']['images_dir']
        
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        """This method is called by Lightning before training/testing/validation."""
        self.train_dataset = datasets.ImageFolder(
            os.path.join(self.images_dir, 'train'),
            transform=self.transform
        )
        self.val_dataset = datasets.ImageFolder(
            os.path.join(self.images_dir, 'val'),
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        pass
