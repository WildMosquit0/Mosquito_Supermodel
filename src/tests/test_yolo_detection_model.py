import torch
import pytest
from src.models.yolo_detection_model import YOLODetectionModel

@pytest.fixture
def dummy_detection_model():
    # Load YOLOv8 from ultralytics for object detection
    return YOLODetectionModel(model_path='yolov8n.pt')

@pytest.fixture
def dummy_batch():
    # Dummy batch with random tensors simulating images
    images = torch.randn(4, 3, 256, 256)  # Batch of 4 images, 3 channels, 256x256
    targets = torch.randn(4, 5)  # Dummy targets
    return {'images': images, 'targets': targets}

def test_forward(dummy_detection_model, dummy_batch):
    outputs = dummy_detection_model(dummy_batch['images'])
    assert outputs is not None  # Ensure output is not None

def test_training_step(dummy_detection_model, dummy_batch):
    loss = dummy_detection_model.training_step(dummy_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)  # Ensure training returns a tensor

def test_validation_step(dummy_detection_model, dummy_batch):
    val_output = dummy_detection_model.validation_step(dummy_batch, batch_idx=0)
    assert 'val_loss' in val_output
    assert isinstance(val_output['val_loss'], torch.Tensor)  # Ensure validation returns a tensor