import cv2
import os
from ultralytics import YOLO
import torch
from typing import List

from src.utils.common import create_output_dir

class Inferer:
    def __init__(self, model_path: str, task: str, output_dir: str, images_dir: str, save_animations: bool=False) -> None:
        self.model = YOLO(model_path)
        self.task = task
        self.output_dir = os.path.join(output_dir, task)
        self.images_dir = images_dir
        self.save_animations = save_animations
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        create_output_dir(self.output_dir)

    def infer(self, persist: bool = True) -> List:
        if self.task == 'track':
            results = self.model.track(source=self.images_dir, persist=persist)
        else:
            results = self.model.predict(source=self.images_dir)
        
        if self.save_animations:
            self.save_animation(results)
        return results

    def save_animation(self, results):
        for idx, result in enumerate(results):
            annotated_img = result.plot()
            save_path = os.path.join(self.output_dir, f"annotated_{idx}.jpg")
            cv2.imwrite(save_path, annotated_img)