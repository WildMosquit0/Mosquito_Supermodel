import torch
from sklearn.metrics import precision_score, recall_score
from src.utils.box_utils import compute_box_iou

class DetectionMetrics:
    def __init__(self, iou_threshold=0.5):
        """
        Initialize precision, recall, and IoU metrics.
        Args:
            iou_threshold: IoU threshold for matching predicted and ground-truth boxes.
        """
        self.iou_threshold = iou_threshold

    def precision(self, preds, targets):
        """
        Calculate precision using IoU between predicted and target bounding boxes.
        Args:
            preds: Predicted bounding boxes [x1, y1, x2, y2]
            targets: Ground-truth bounding boxes [x1, y1, x2, y2]
        Returns:
            torch.Tensor: Precision score
        """
        if preds.shape[0] == 0 or targets.shape[0] == 0:
            return torch.tensor(0.0)

        iou_matrix = compute_box_iou(preds, targets)
        # True positives: IoU > threshold
        matches = (iou_matrix > self.iou_threshold).sum(dim=1).clamp(0, 1).cpu().numpy()

        # Precision: using sklearn for simplicity
        precision = precision_score([1] * len(targets), matches)
        return torch.tensor(precision)

    def recall(self, preds, targets):
        """
        Calculate recall using IoU between predicted and target bounding boxes.
        Args:
            preds: Predicted bounding boxes [x1, y1, x2, y2]
            targets: Ground-truth bounding boxes [x1, y1, x2, y2]
        Returns:
            torch.Tensor: Recall score
        """
        if preds.shape[0] == 0 or targets.shape[0] == 0:
            return torch.tensor(0.0)

        iou_matrix = compute_box_iou(preds, targets)
        # True positives: IoU > threshold
        matches = (iou_matrix > self.iou_threshold).sum(dim=0).clamp(0, 1).cpu().numpy()

        # Recall: using sklearn for simplicity
        recall = recall_score([1] * len(targets), matches)
        return torch.tensor(recall)

    def compute_iou(self, preds, targets):
        """
        Calculate IoU between predicted and ground truth boxes using the refactored function.
        Args:
            preds: Predicted bounding boxes [x1, y1, x2, y2]
            targets: Ground-truth bounding boxes [x1, y1, x2, y2]
        Returns:
            torch.Tensor: IoU matrix between each predicted and target box.
        """
        return compute_box_iou(preds, targets)
