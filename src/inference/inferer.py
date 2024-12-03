import cv2
import os
import json
from ultralytics import YOLO
import torch
from typing import List, Dict
from src.utils.common import create_output_dir

class Inferer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        
        model_path = config['model']['weights']
        self.task = config['model']['task']
        self.conf_threshold = config['model'].get('conf_threshold', 0.5)  # Confidence threshold default is 0.5
        self.iou_threshold = config['model'].get('iou_threshold', 0.45)  # IoU threshold default is 0.45
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, else CPU

        self.output_dir = os.path.join(config['output']['output_dir'], self.task)
        self.images_dir = config['input']['images_dir']
        self.save_animations = config['output'].get('save_animations', False)
        self.model = YOLO(model_path)
        self.model.to(self.device)

        create_output_dir(self.output_dir)

    def infer(self, persist: bool = True) -> List:
        if self.task == 'track':
            results = self.model.track(
                source=self.images_dir, 
                conf=self.conf_threshold, 
                iou=self.iou_threshold,   
                persist=persist,
                save=self.save_animations,
                name = self.images_dir
            )
        else:
            results = self.model.predict(
                source=self.images_dir, 
                conf=self.conf_threshold,  
                iou=self.iou_threshold,
                save=self.save_animations    
            )
        
        if self.save_animations:
            self.save_animation(results)
        return results

    def save_animation(self, results):
        for idx, result in enumerate(results):
            
            annotated_img = result.plot()
            image_name = os.path.basename(result.path)
            save_path = os.path.join(self.output_dir, image_name)
            cv2.imwrite(save_path, annotated_img)

if __name__ == "__main__":
    inferer = Inferer(config_path='config.json')

    results = inferer.infer()
