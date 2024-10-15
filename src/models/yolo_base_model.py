import torch
import pytorch_lightning as pl
from abc import ABC, abstractmethod

class YOLOBaseModel(pl.LightningModule, ABC):
    def __init__(self, config):
        super(YOLOBaseModel, self).__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config  # Store the configuration
        self.learning_rate = config['training'].get('lr', 1e-3)  # Get learning rate from config
        self.optimizer_type = config['training'].get('optimizer', 'adam').lower()  # Optimizer type (default to 'adam')

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
        """
        Configures the optimizer based on the config file.
        Supports Adam and SGD optimizers, more can be added as needed.
        """
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        return [optimizer], [scheduler]
