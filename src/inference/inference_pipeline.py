from src.models.yolo_detection_model import YOLODetectionModel

def inference(image_path: str):
    model = YOLODetectionModel(model_path='yolov8.pt')
    results = model([image_path])  # Perform inference on a single image
    return results