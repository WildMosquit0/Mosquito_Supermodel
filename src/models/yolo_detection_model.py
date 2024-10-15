import torch
from ultralytics import YOLO
from src.models.yolo_base_model import YOLOBaseModel
from src.utils.loss_functions import compute_ciou_loss
from src.utils.metrics import DetectionMetrics

class YOLODetectionModel(YOLOBaseModel):
    def __init__(self, config: dict, model: torch.nn.Module = None):
        # Initialize the base YOLO model with the config
        super(YOLODetectionModel, self).__init__(config)
        
        self.task = config['model'].get('task', 'detect')
        self.conf_threshold = config['model'].get('conf_threshold', 0.1)
        self.ciou_threshold = config['model'].get('iou_threshold', 0.45)  # Use the cIoU threshold for both metrics and loss

        # Load the YOLO model from the provided path or directly if passed as a model
        if model is not None:
            self.model = model
        else:
            model_path = config['model'].get('weights')
            if model_path is not None:
                self.model = YOLO(model_path)
            else:
                raise ValueError("Either 'model' or 'model_path' must be provided.")
        
        # Move the model to the appropriate device (GPU or CPU)
        self.model.to(self._device)

        # Initialize custom metrics for tracking performance (cIoU, precision, recall)
        self.metrics = DetectionMetrics(iou_threshold=self.ciou_threshold)  # Ensure cIoU threshold is consistent

    def forward(self, x):
        # Forward pass through the YOLO model
        x = x.to(self._device)
        if self.task == 'detect':
            outputs = self.model.predict(x, conf=self.conf_threshold, verbose=False)
        else:
            outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        # Extract images and targets from the batch
        images = batch['images'].to(self._device)
        targets = batch['targets'].to(self._device)
        
        # Perform a forward pass and compute loss
        results = self.model(images, targets)
        train_loss = compute_ciou_loss(results.boxes, targets)
        
        # Compute metrics
        precision = self.metrics.precision(results.boxes, targets)
        recall = self.metrics.recall(results.boxes, targets)
        ciou = self.metrics.compute_ciou(results.boxes, targets)

        # Log metrics
        self.log('train_loss', train_loss, on_step=True, on_epoch=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True)
        self.log('train_ciou', ciou.mean(), on_step=True, on_epoch=True)
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        # Extract images and targets from the validation batch
        images = batch['images'].to(self._device)
        targets = batch['targets'].to(self._device)
        
        # Perform a forward pass and compute validation loss
        results = self.model(images, targets)
        val_loss = compute_ciou_loss(results.boxes, targets)

        # Compute metrics
        precision = self.metrics.precision(results.boxes, targets)
        recall = self.metrics.recall(results.boxes, targets)
        ciou = self.metrics.compute_ciou(results.boxes, targets)

        # Log validation metrics
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_precision', precision, on_epoch=True)
        self.log('val_recall', recall, on_epoch=True)
        self.log('val_ciou', ciou.mean(), on_epoch=True)
        
        return val_loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler based on the config.
        Supports Adam and SGD optimizers, and step-based learning rate scheduling.
        """
        # Call the optimizer configuration from the base model (config-driven)
        return super().configure_optimizers()
