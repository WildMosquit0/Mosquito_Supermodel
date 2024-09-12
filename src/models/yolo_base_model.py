import torch
import pytorch_lightning as pl
from typing import Any

class YOLOBaseModel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, num_classes: int):
        super(YOLOBaseModel, self).__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
