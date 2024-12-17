import os
from ultralytics import YOLO

root = '/home/bohbot/Evyatar'
model_path = 'runs/detect/most_update_with_insects_250_epocs4/weights/best.pt'
data='yaml/val.yaml'

model = YOLO(os.path.join(root, model_path))

validation_results = model.val(
    data=os.path.join(root, data), 
    imgsz=640, 
    batch=16, 
    conf=0.25, 
    iou=0.6, 
    device="0")

