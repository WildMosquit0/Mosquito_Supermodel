# test_yolo_tracking_model.py

import pytest
import torch
from src.models.yolo_tracking_model import YOLOTrackingModel
import torch.nn as nn

class MockTrackingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def track(self, x, verbose=False):
        return [{'boxes': torch.randn(2, 4),
                 'scores': torch.randn(2),
                 'labels': torch.tensor([1, 2]),
                 'ids': torch.tensor([101, 102])}]

    def __call__(self, x, tracks=None):
        class Results:
            def __init__(self):
                self.loss = torch.tensor(0.5, requires_grad=True)
        return Results()

@pytest.fixture
def dummy_tracking_model():
    mock_model = MockTrackingModel()
    model = YOLOTrackingModel(model=mock_model, task='track')
    return model

def test_forward(dummy_tracking_model):
    x = torch.randn(1, 10)
    outputs = dummy_tracking_model(x)
    assert isinstance(outputs, list)
    for output in outputs:
        assert 'boxes' in output and 'scores' in output and 'labels' in output and 'ids' in output

def test_training_step(dummy_tracking_model):
    # Adjusted to set task='train' to use training_step
    dummy_tracking_model.task = 'train'
    batch = {'images': torch.randn(4, 10), 'tracks': None}
    loss = dummy_tracking_model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(loss)
    assert loss.item() == 0.5

def test_validation_step(dummy_tracking_model):
    # Adjusted to set task='train' to use validation_step
    dummy_tracking_model.task = 'train'
    batch = {'images': torch.randn(4, 10), 'tracks': None}
    val_loss = dummy_tracking_model.validation_step(batch, batch_idx=0)
    assert torch.is_tensor(val_loss)
    assert val_loss.item() == 0.5

def test_configure_optimizers(dummy_tracking_model):
    optimizer = dummy_tracking_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups[0]['params']) > 0
