from typing import List
import cv2
import os
from ultralytics import YOLO
import torch

class Inferer:
    def __init__(self, model_path: str, task: str = 'detect', output_dir: str = './output'):
        # Initialize the YOLO model based on the task
        self.model = YOLO(model_path)
        self.task = task
        self.output_dir = output_dir  # Custom output directory
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to device
        os.makedirs(self.output_dir, exist_ok=True)

        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def infer(self, image_paths: List[str], persist=True):
        results = []
        for image_path in image_paths:
            if self.task == 'track':
                result = self.model.track(source=image_path, persist=persist)
            else:
                result = self.model.predict(source=image_path)

            # Access detection boxes, scores, and labels from tracking output
            boxes = result[0].boxes
            scores = boxes.conf
            labels = boxes.cls
            track_ids = boxes.id if self.task == 'track' else None  # Get track IDs only for tracking

            # Print results
            print(f"Boxes: {boxes}")
            print(f"Scores: {scores}")
            print(f"Labels: {labels}")
            if track_ids is not None:
                print(f"Track IDs: {track_ids}")

            # Save the annotated image in the custom output directory
            annotated_img = result[0].plot()  # Annotated image with boxes
            save_path = os.path.join(self.output_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, annotated_img)  # Save the annotated image

            results.append(result)
        return results

if __name__ == "__main__":
    model_path = 'yolov8n.pt'  # Path to your YOLOv8 model
    inferer = Inferer(model_path=model_path, task='track', output_dir='./tracking_output')  # Use 'track' or 'detect', and specify custom output directory

    image_paths = ['src/tests/images/0.jpg', 'src/tests/images/1.jpg']

    # Run inference (tracking or detection) on images
    results = inferer.infer(image_paths=image_paths)

    for idx, result in enumerate(results):
        print(f"Result for image {idx}:")
        print(result)
