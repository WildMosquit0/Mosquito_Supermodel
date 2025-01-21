import cv2
import os
import json
from ultralytics import YOLO
import torch
from typing import List, Dict
import subprocess
from src.utils.common import create_output_dir


def ensure_weights():
    try:
        subprocess.run(["git", "lfs", "pull"], check=True)
        print("Weights downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download weights: {e}")



class Inferer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        ensure_weights()

        model_path = config['model']['weights']
        self.task = config['model']['task']
        self.conf_threshold = config['model'].get('conf_threshold', 0.5) 
        self.iou_threshold = config['model'].get('iou_threshold', 0.45)  
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  

        self.output_dir = config['output_dir']
        self.images_dir = config['images_dir']
        self.save_animations = config.get('save_animations', False)
        self.vid_stride = config['model'].get('vid_stride', 1)
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
                vid_stride=self.vid_stride,
                project=self.output_dir,
                exist_ok=True
            )
        else:
            results = self.model.predict(
                source=self.images_dir, 
                conf=self.conf_threshold,  
                iou=self.iou_threshold,
                save=self.save_animations,
                vid_stride=self.vid_stride,
                project=self.output_dir,
                exist_ok=True    
            )
        
        if self.save_animations:
            self.save_animation(results)

        print(f"Results saved to: {self.output_dir}")
        return results


    def save_animation(self, results):
        for idx, result in enumerate(results):
            
            annotated_img = result.plot()
            image_name = os.path.basename(result.path)
            save_path = os.path.join(self.output_dir, image_name)
            #cv2.imwrite(save_path, annotated_img)