import torch
import torch.nn as nn
import pytest
from src.models.yolo_detection_model import YOLODetectionModel
from src.models.yolo_tracking_model import YOLOTrackingModel

# Simple CNN model to replace Linear layers
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 64 * 64, 5)  # Output will be compatible with (batch_size, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        x = self.fc(x)
        return x

@pytest.fixture
def dummy_detection_model():
    model = SimpleCNN()  # Use a CNN model that can handle image input
    return YOLODetectionModel(model=model, num_classes=5)

@pytest.fixture
def dummy_tracking_model():
    model = SimpleCNN()  # Use the same simple CNN for tracking
    return YOLOTrackingModel(model=model, num_classes=5)

@pytest.fixture
def dummy_batch():
    images = torch.randn(4, 3, 256, 256)  # Batch of 4 images, 3 channels, 256x256
    targets = torch.randn(4, 5)  # Corresponding targets for detection
    tracks = torch.randn(4, 5)  # Corresponding tracking data
    return {'images': images, 'targets': targets, 'tracks': tracks}