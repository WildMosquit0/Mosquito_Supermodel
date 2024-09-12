from pytorch_lightning import Trainer
from src.models.yolo_detection_model import YOLODetectionModel
from src.data.data_loader import get_dataloader

def train():
    model = YOLODetectionModel(model_path='yolov8.pt')  # Using pretrained YOLOv8
    trainer = Trainer(max_epochs=10, gpus=1)
    train_loader, val_loader = get_dataloader()  # Assume this loads your data
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)