# test_yolo_base_model.py

import pytest
import torch
from src.models.yolo_base_model import YOLOBaseModel
import torch.nn as nn

class ConcreteYOLOModel(YOLOBaseModel):
    def __init__(self):
        # Use a simple model with parameters
        dummy_model = nn.Linear(10, 5)  # Input features: 10, Output features: 5
        super().__init__(dummy_model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = torch.tensor(0.0)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = torch.tensor(0.0)
        return val_loss

def test_configure_optimizers():
    model = ConcreteYOLOModel()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups[0]['params']) > 0, "Optimizer should have parameters."
