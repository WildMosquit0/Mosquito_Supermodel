import torch
import pytorch_lightning as pl
from abc import ABC, abstractmethod

class YOLOBaseModel(pl.LightningModule, ABC):
    def __init__(self):
        super(YOLOBaseModel, self).__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
