# test_yolo_detection_model.py

import pytest
import torch
from src.models.yolo_detection_model import YOLODetectionModel
import torch.nn as nn

class MockDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def predict(self, x, verbose=False):
        return [{'boxes': torch.randn(2, 4),
                 'scores': torch.randn(2),
                 'labels': torch.tensor([1, 2])}]

    def __call__(self, x, targets=None):
        class Results:
            def __init__(self):
                self.loss = torch.tensor(0.5, requires_grad=True)
        return Results()

@pytest.fixture
def dummy_detection_model():
    mock_model = MockDetectionModel()
    model = YOLODetectionModel(model=mock_model, task='detect')
    return model

def test_forward(dummy_detection_model):
    x = torch.randn(1, 10)
    outputs = dummy_detection_model(x)
    assert isinstance(outputs, list)
    for output in outputs:
        assert 'boxes' in output and 'scores' in output and 'labels' in output

def test_training_step(dummy_detection_model):
    # Adjusted to set task='train' to use training_step
    dummy_detection_model.task = 'train'
    batch = {'images': torch.randn(4, 10), 'targets': None}
    loss = dummy_detection_model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(loss)
    assert loss.item() == 0.5

def test_validation_step(dummy_detection_model):
    # Adjusted to set task='train' to use validation_step
    dummy_detection_model.task = 'train'
    batch = {'images': torch.randn(4, 10), 'targets': None}
    val_loss = dummy_detection_model.validation_step(batch, batch_idx=0)
    assert torch.is_tensor(val_loss)
    assert val_loss.item() == 0.5

def test_configure_optimizers(dummy_detection_model):
    optimizer = dummy_detection_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups[0]['params']) > 0
