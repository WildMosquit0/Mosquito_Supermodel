import torch
from ultralytics import YOLO
from src.models.yolo_base_model import YOLOBaseModel
from src.utils.loss_functions import compute_ciou_loss
from src.utils.metrics import DetectionMetrics
from src.config.overrides import YOLOOverrides

class YOLODetectionModel(YOLOBaseModel):
    """
    YOLODetectionModel handles training and validation for YOLO-based object detection models.
    """
    
    def __init__(self, config: dict):
        super(YOLODetectionModel, self).__init__(config)
        self.config = config
        self.task = config['model'].get('task', 'detect')
        model_path = config['model']['weights']
        
        if not isinstance(model_path, str):
            raise TypeError(f"Expected model path to be a string, but got {type(model_path)}")
        
        self.model = YOLO(model_path)
        
        # Apply overrides
        overrides = YOLOOverrides(config)
        self.model.overrides.update(overrides.get_overrides())

        self.model.eval()
        self.model.to(self._device)
        self.metrics = DetectionMetrics(iou_threshold=config['model'].get('iou_threshold', 0.5))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if self.task == 'detect':
            outputs = self.model.predict(x, verbose=False)
        else:
            outputs = self.model(x)
        return outputs

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, targets = batch
        self.model.train()
        results = self.model(images)
        results.boxes.requires_grad_(True)
        train_loss = compute_ciou_loss(results.boxes, targets)
        self.log('train_loss', train_loss)
        precision = self.metrics.precision(results.boxes, targets)
        recall = self.metrics.recall(results.boxes, targets)
        self.log('train_precision', precision, on_step=True, on_epoch=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True)
        return train_loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, targets = batch
        self.model.eval()
        results = self.model(images)
        val_loss = compute_ciou_loss(results.boxes, targets)
        self.log('val_loss', val_loss)
        precision = self.metrics.precision(results.boxes, targets)
        recall = self.metrics.recall(results.boxes, targets)
        self.log('val_precision', precision, on_step=True, on_epoch=True)
        self.log('val_recall', recall, on_step=True, on_epoch=True)
        return val_loss
