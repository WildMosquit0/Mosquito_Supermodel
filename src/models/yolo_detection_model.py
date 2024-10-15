import torch
import pytorch_lightning as pl
from ultralytics import YOLO
from src.models.yolo_base_model import YOLOBaseModel
from src.utils.loss_functions import compute_ciou_loss
from src.utils.metrics import DetectionMetrics

class YOLODetectionModel(YOLOBaseModel):
    def __init__(self, config: dict, model: torch.nn.Module = None):
        super(YOLODetectionModel, self).__init__()
        
        self.config = config
        model_path = config['model'].get('weights')
        self.task = config['model'].get('task', 'detect')
        self.conf_threshold = config['model'].get('conf_threshold', 0.1)
        self.iou_threshold = config['model'].get('iou_threshold', 0.45)
        
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = YOLO(model_path)
        else:
            raise ValueError("Either model or model_path must be provided.")
        
        self.model.to(self._device)
        
        self.metrics = DetectionMetrics(num_classes=1, iou_threshold=self.iou_threshold)

    def forward(self, x):
        x = x.to(self._device)
        if self.task == 'detect':
            outputs = self.model.predict(x, conf=self.conf_threshold, verbose=False)
        else:
            outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        images = batch['images'].to(self._device)
        targets = batch['targets'].to(self._device)
        
        results = self.model(images, targets)
        
        train_loss = compute_ciou_loss(results.boxes, targets)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True)
        
        precision = self.metrics.precision(results.boxes, targets)
        recall = self.metrics.recall(results.boxes, targets)
        iou = self.metrics.iou(results.boxes, targets)
        
        self.log('train_precision', precision, on_step=True, on_epoch=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True)
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        images = batch['images'].to(self._device)
        targets = batch['targets'].to(self._device)
        
        results = self.model(images, targets)
        
        val_loss = compute_ciou_loss(results.boxes, targets)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        
        precision = self.metrics.precision(results.boxes, targets)
        recall = self.metrics.recall(results.boxes, targets)
        iou = self.metrics.iou(results.boxes, targets)
        
        self.log('val_precision', precision, on_epoch=True)
        self.log('val_recall', recall, on_epoch=True)
        self.log('val_iou', iou, on_epoch=True)
        
        return val_loss

    def configure_optimizers(self):
        lr = self.config['training'].get('lr', 1e-3)
        optimizer_name = self.config['training'].get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]
