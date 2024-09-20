# test_yolo_detection_model.py

import pytest
import torch
from unittest.mock import MagicMock, patch
from src.models.yolo_detection_model import YOLODetectionModel
import torch.nn as nn

import torch
import torch.nn as nn


class MockResult:
    def __init__(self):
        self.pred = [{'boxes': torch.randn(2, 4),
                      'scores': torch.randn(2),
                      'labels': torch.tensor([1, 2]),
                      'ids': torch.tensor([101, 102])}]
        self.loss = torch.tensor(0.5, requires_grad=True)


class MockDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)  # Simple model with parameters
    
    def forward(self, x, targets=None):
        return MockResult()


@pytest.fixture
def dummy_detection_model():
    mock_model = MockDetectionModel()
    model = YOLODetectionModel(model=mock_model)
    return model

def test_configure_optimizers(dummy_detection_model):
    optimizer = dummy_detection_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups[0]['params']) > 0, "Optimizer should have parameters."
