# src/models/yolo_detection_model.py

import torch
from ultralytics import YOLO
from src.models.yolo_base_model import YOLOBaseModel

class YOLODetectionModel(YOLOBaseModel):
    def __init__(self, model_path: str = None, task: str = 'detect', model: torch.nn.Module = None):
        super(YOLODetectionModel, self).__init__()
        self.task = task
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = YOLO(model_path)
        else:
            raise ValueError("Either model or model_path must be provided.")

        self.model.to(self._device)

    def forward(self, x):
        x = x.to(self.device)
        if self.task == 'detect':
            outputs = self.model.predict(x, verbose=False)
        else:
            outputs = self.model(x)
        return outputs
    
    def predict(self, source: str):
        return self.model.predict(source)

    def training_step(self, batch, batch_idx):
        images = batch['images']
        targets = batch['targets']
        results = self.model(images, targets)
        loss = results.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['images']
        targets = batch['targets']
        results = self.model(images, targets)
        val_loss = results.loss
        self.log('val_loss', val_loss)
        return val_loss
