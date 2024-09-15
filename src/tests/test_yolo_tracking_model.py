import pytest
from PIL import Image
from ultralytics import YOLO

@pytest.fixture
def dummy_tracking_model():
    # Load YOLOv8 from ultralytics for tracking
    return YOLO("yolov8n.pt")

def test_forward(dummy_tracking_model):
    # Load the image
    img_path = "src/tests/images/0.jpg"
    
    # Run forward pass with save=True to save the annotated image
    results = dummy_tracking_model.predict(source=img_path, save=True, save_txt=True)

    # Assert the results are not None
    assert results is not None
