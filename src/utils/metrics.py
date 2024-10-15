from torchmetrics import Precision, Recall
import torch
import torch.nn.functional as F

class DetectionMetrics:
    def __init__(self, num_classes=1, iou_threshold=0.5):
        """
        Initialize precision, recall, and IoU metrics.
        Args:
            num_classes: Number of classes for precision/recall
            iou_threshold: IoU threshold for computing precision and recall
        """
        self.precision_metric = Precision(num_classes=num_classes, average='macro', threshold=iou_threshold)
        self.recall_metric = Recall(num_classes=num_classes, average='macro', threshold=iou_threshold)
        self.iou_threshold = iou_threshold

    def precision(self, preds, targets):
        """
        Calculate precision for given predictions and targets.
        Args:
            preds: Predicted bounding boxes
            targets: Ground-truth bounding boxes
        Returns:
            torch.Tensor: Precision score
        """
        return self.precision_metric(preds, targets)

    def recall(self, preds, targets):
        """
        Calculate recall for given predictions and targets.
        Args:
            preds: Predicted bounding boxes
            targets: Ground-truth bounding boxes
        Returns:
            torch.Tensor: Recall score
        """
        return self.recall_metric(preds, targets)

    def iou(self, preds, targets):
        """
        Calculate Intersection over Union (IoU) between predicted and ground truth boxes.
        Args:
            preds: Predicted bounding boxes
            targets: Ground-truth bounding boxes
        Returns:
            torch.Tensor: IoU score
        """
        return self.compute_iou(preds, targets)

    def compute_iou(self, preds, targets):
        """
        Helper function to compute IoU.
        Args:
            preds: Predicted bounding boxes
            targets: Ground-truth bounding boxes
        Returns:
            torch.Tensor: IoU scores
        """
        # Calculate intersection
        inter_x1 = torch.max(preds[:, 0], targets[:, 0])
        inter_y1 = torch.max(preds[:, 1], targets[:, 1])
        inter_x2 = torch.min(preds[:, 2], targets[:, 2])
        inter_y2 = torch.min(preds[:, 3], targets[:, 3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Calculate union
        pred_area = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
        target_area = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
        union_area = pred_area + target_area - inter_area

        return inter_area / union_area
