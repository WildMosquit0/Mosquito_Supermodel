# src/models/yolo_tracking_model.py

import torch
from ultralytics import YOLO
from src.models.yolo_base_model import YOLOBaseModel

class YOLOTrackingModel(YOLOBaseModel):
    def __init__(self, model_path: str = None, task: str = 'track', model: torch.nn.Module = None):
        super(YOLOTrackingModel, self).__init__()
        self.task = task
        if model is not None:
            self.model = model
        elif model_path is not None:
            if self.task == 'track':
                # Initialize YOLO model for tracking inference
                self.model = YOLO(model_path, task=self.task)
            else:
                # Initialize YOLO model for training
                self.model = YOLO(model_path)
        else:
            raise ValueError("Either model or model_path must be provided.")

    def forward(self, x):
        if self.task == 'track':
            # Use the 'track' method for inference
            outputs = self.model.track(x, verbose=False)
        else:
            # Use the model directly for training
            outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        images = batch['images']
        tracks = batch['tracks']
        results = self.model(images, tracks)
        loss = results.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['images']
        tracks = batch['tracks']
        results = self.model(images, tracks)
        val_loss = results.loss
        self.log('val_loss', val_loss)
        return val_loss
