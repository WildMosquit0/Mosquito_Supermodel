import cv2
import os
from ultralytics import YOLO
import torch
from typing import List
from src.utils.config import load_config

class Inferer:
    def __init__(self, config: dict):
        self.model = YOLO(config["model_path"])
        self.task = config["task"]
        self.output_dir = config["output_dir"]
        self.images_dir = config["images_dir"]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def infer(self, persist: bool = True) -> List:
        if self.task == 'track':
            results = self.model.track(source=self.images_dir, persist=persist)
        else:
            results = self.model.predict(source=self.images_dir)

        for idx, result in enumerate(results):
            annotated_img = result.plot()
            save_path = os.path.join(self.output_dir, f"annotated_{idx}.jpg")
            cv2.imwrite(save_path, annotated_img)

        return results

if __name__ == "__main__":
    config = load_config("config.json")
    inferer = Inferer(config)
    results = inferer.infer()

    for idx, result in enumerate(results):
        print(f"Result for image {idx}: {result}")
