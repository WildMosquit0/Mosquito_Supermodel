import torch
from ultralytics import YOLO
from src.models.yolo_base_model import YOLOBaseModel
from src.utils.loss_functions import compute_ciou_loss
from src.utils.metrics import DetectionMetrics

class YOLODetectionModel(YOLOBaseModel):
    def __init__(self, config):
        super(YOLODetectionModel, self).__init__(config)
        self.config = config
        self.task = config['model'].get('task', 'detect')

        # Load YOLO model weights without triggering any internal training logic
        model_path = config['model'].get('weights')
        if model_path:
            self.model = YOLO(model_path)
        else:
            raise ValueError("No model weights specified in the config.")

        # Disable Ultralytics' internal data handling and training logic
        self.model.overrides['data'] = ''  # Ensure no data loading internally
        self.model.overrides['epochs'] = config['training'].get('epochs', 10)  # Ensure epochs come from config
        self.model.eval()  # Set model to eval mode by default

        # Move model to the correct device
        self.model.to(self._device)

        # Set up detection metrics
        self.metrics = DetectionMetrics(iou_threshold=config['model'].get('iou_threshold', 0.5))

    def forward(self, x):
        x = x.to(self.device)
        if self.task == 'detect':
            outputs = self.model.predict(x, verbose=False)
        else:
            outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # Make predictions
        results = self.model(images)

        # Ensure results have requires_grad=True
        results.boxes.requires_grad_(True)

        # Compute cIoU loss
        train_loss = compute_ciou_loss(results.boxes, targets)
        self.log('train_loss', train_loss)

        # Calculate metrics
        precision = self.metrics.precision(results.boxes, targets)
        recall = self.metrics.recall(results.boxes, targets)

        self.log('train_precision', precision, on_step=True, on_epoch=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # Make predictions
        results = self.model(images)

        # Compute cIoU loss
        val_loss = compute_ciou_loss(results.boxes, targets)
        self.log('val_loss', val_loss)

        # Calculate metrics
        precision = self.metrics.precision(results.boxes, targets)
        recall = self.metrics.recall(results.boxes, targets)

        self.log('val_precision', precision, on_step=True, on_epoch=True)
        self.log('val_recall', recall, on_step=True, on_epoch=True)

        return val_loss
