# test_inference_detection_tracking.py

import pytest
import torch
from src.models.yolo_detection_model import YOLODetectionModel
from src.models.yolo_tracking_model import YOLOTrackingModel
from torchvision import transforms
from PIL import Image
import os

IMAGES_DIR = 'src/tests/images'

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

@pytest.fixture(scope='session')
def image_tensors():
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')])
    images = []
    for file_name in image_files:
        image_path = os.path.join(IMAGES_DIR, file_name)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        images.append(image_tensor)
    images_tensor = torch.stack(images)  # Shape: (batch_size, 3, 320, 320)
    images_tensor = images_tensor.to(device)
    return images_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture(scope='session')
def detection_model():
    # Instantiate the detection model with task='detect' for inference
    model = YOLODetectionModel(model_path='yolov8n.pt', task='detect')
    model.eval()
    model.to(device) 
    return model

@pytest.fixture(scope='session')
def tracking_model():
    # Instantiate the tracking model with task='track' for inference
    model = YOLOTrackingModel(model_path='yolov8n.pt', task='track')
    model.eval()
    model.to(device) 
    return model

def test_inference_detection(detection_model, image_tensors):
    with torch.no_grad():
        # Run detection using the forward method
        detection_outputs = detection_model(image_tensors)

    # Assertions and printing outputs...
    assert isinstance(detection_outputs, list), "Detection outputs should be a list."
    for output in detection_outputs:
        assert 'boxes' in output and 'scores' in output and 'labels' in output
        assert output['boxes'].shape[1] == 4, "Each box should have 4 coordinates."

    # Print detection outputs
    print("Detection outputs:")
    for idx, output in enumerate(detection_outputs):
        print(f"Image {idx}:")
        print(f"Boxes: {output['boxes']}")
        print(f"Scores: {output['scores']}")
        print(f"Labels: {output['labels']}")

def test_inference_tracking(tracking_model, image_tensors):
    with torch.no_grad():
        # Run tracking using the forward method
        tracking_outputs = tracking_model(image_tensors)

    # Assertions and printing outputs...
    assert isinstance(tracking_outputs, list), "Tracking outputs should be a list."
    for output in tracking_outputs:
        assert 'boxes' in output and 'scores' in output and 'labels' in output and 'ids' in output
        assert output['boxes'].shape[1] == 4, "Each box should have 4 coordinates."

    # Print tracking outputs
    print("Tracking outputs:")
    for idx, output in enumerate(tracking_outputs):
        print(f"Image {idx}:")
        print(f"Boxes: {output['boxes']}")
        print(f"Scores: {output['scores']}")
        print(f"Labels: {output['labels']}")
        print(f"IDs: {output['ids']}")
