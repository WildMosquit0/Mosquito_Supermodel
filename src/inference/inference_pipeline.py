from typing import List

from src.models.yolo_detection_model import YOLODetectionModel


class Inferer:
    def __init__(self, model_path: str):
        self.model = YOLODetectionModel(model_path=model_path)

    def infer(self, image_paths: List[str]):
        results = []
        for image_path in image_paths:
            result = self.model.predict(source=image_path)
            results.append(result)
        return results

    def infer_image(self, image_path: str):
        results = self.model.predict(source=image_path)
        return results
