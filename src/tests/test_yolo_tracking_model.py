import pytest
import torch
from unittest.mock import MagicMock, patch
from src.models.yolo_tracking_model import YOLOTrackingModel

import torch
import torch.nn as nn


class MockResult:
    def __init__(self):
        self.pred = [{'boxes': torch.randn(2, 4),
                      'scores': torch.randn(2),
                      'labels': torch.tensor([1, 2]),
                      'ids': torch.tensor([101, 102])}]
        self.loss = torch.tensor(0.5, requires_grad=True)

class MockTrackingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)  # Simple model with parameters
    
    def track(self, x, tracks=None):
        return MockResult()

@pytest.fixture
def dummy_tracking_model():
    mock_model = MockTrackingModel()
    model = YOLOTrackingModel(model=mock_model)
    return model

def test_configure_optimizers(dummy_tracking_model):
    optimizer = dummy_tracking_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups[0]['params']) > 0, "Optimizer should have parameters."
