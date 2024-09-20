import torch
from ultralytics import YOLO
from src.models.yolo_base_model import YOLOBaseModel

class YOLOTrackingModel(YOLOBaseModel):
    def __init__(self, model=None, model_path=None):
        if model is not None:
            super().__init__(model)
        else:
            model = YOLO(model_path)
            super().__init__(model)

    def forward(self, x):
        outputs = self.model.track(x)
        return outputs.pred

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
