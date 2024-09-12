import torch
import pytest
from src.models.yolo_tracking_model import YOLOTrackingModel

@pytest.fixture
def dummy_tracking_model():
    # Load YOLOv8 from ultralytics for tracking
    return YOLOTrackingModel(model_path='yolov8n.pt')

@pytest.fixture
def dummy_batch():
    # Dummy batch with random tensors simulating images and tracks
    images = torch.randn(4, 3, 256, 256)  # Batch of 4 images, 3 channels, 256x256
    tracks = torch.randn(4, 5)  # Dummy tracks for tracking task
    return {'images': images, 'tracks': tracks}

def test_forward(dummy_tracking_model, dummy_batch):
    outputs = dummy_tracking_model(dummy_batch['images'])
    assert outputs is not None  # Ensure output is not None

def test_training_step(dummy_tracking_model, dummy_batch):
    loss = dummy_tracking_model.training_step(dummy_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)  # Ensure training returns a tensor

def test_validation_step(dummy_tracking_model, dummy_batch):
    val_output = dummy_tracking_model.validation_step(dummy_batch, batch_idx=0)
    assert 'val_loss' in val_output
    assert isinstance(val_output['val_loss'], torch.Tensor)  # Ensure validation returns a tensor