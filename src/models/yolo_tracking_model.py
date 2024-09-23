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
            self.model = YOLO(model_path)  # Initialize the YOLOv8 model
        else:
            raise ValueError("Either model or model_path must be provided.")

        self.model.to(self._device)  # Move model to the device

    def forward(self, x, persist=True):
        x = x.to(self._device)  # Ensure input is on the correct device
        if self.task == 'track':
            # Use the `track()` method for tracking
            outputs = self.model.track(source=x, persist=persist)
        else:
            outputs = self.model(x)
        return outputs
    
    def training_step(self, batch, batch_idx):
        images = batch['images'].to(self._device)
        tracks = batch['tracks']
        results = self.model(images, tracks)
        loss = results.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['images'].to(self._device)
        tracks = batch['tracks']
        results = self.model(images, tracks)
        val_loss = results.loss
        self.log('val_loss', val_loss)
        return val_loss
