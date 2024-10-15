import torch

class DetectionMetrics:
    def __init__(self, iou_threshold=0.5):
        """
        Initialize precision, recall, and IoU metrics using cIoU.
        Args:
            iou_threshold: cIoU threshold for matching predicted and ground-truth boxes
        """
        self.iou_threshold = iou_threshold

    def precision(self, preds, targets):
        """
        Calculate precision using cIoU between predicted and target bounding boxes.
        Args:
            preds: Predicted bounding boxes [x1, y1, x2, y2]
            targets: Ground-truth bounding boxes [x1, y1, x2, y2]
        Returns:
            torch.Tensor: Precision score
        """
        ciou = self.compute_ciou(preds, targets)
        true_positives = (ciou > self.iou_threshold).sum().float()
        predicted_positives = preds.shape[0]  # Number of predicted boxes

        precision = true_positives / (predicted_positives + 1e-7)
        return precision

    def recall(self, preds, targets):
        """
        Calculate recall using cIoU between predicted and target bounding boxes.
        Args:
            preds: Predicted bounding boxes [x1, y1, x2, y2]
            targets: Ground-truth bounding boxes [x1, y1, x2, y2]
        Returns:
            torch.Tensor: Recall score
        """
        ciou = self.compute_ciou(preds, targets)
        true_positives = (ciou > self.iou_threshold).sum().float()
        actual_positives = targets.shape[0]  # Number of ground-truth boxes

        recall = true_positives / (actual_positives + 1e-7)
        return recall

    def compute_ciou(self, preds, targets):
        """
        Compute the Complete IoU (cIoU) between predicted and ground-truth boxes.
        Args:
            preds: Predicted bounding boxes [x1, y1, x2, y2]
            targets: Ground-truth bounding boxes [x1, y1, x2, y2]
        Returns:
            torch.Tensor: cIoU scores between each predicted and target box.
        """
        # Extract coordinates for predicted and target boxes
        pred_x1, pred_y1, pred_x2, pred_y2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        target_x1, target_y1, target_x2, target_y2 = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

        # Compute IoU
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area

        iou = inter_area / (union_area + 1e-7)

        # Compute center distances
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        target_center_x = (target_x1 + target_x2) / 2
        target_center_y = (target_y1 + target_y2) / 2

        center_dist = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2

        # Compute enclosing box
        enclosing_x1 = torch.min(pred_x1, target_x1)
        enclosing_y1 = torch.min(pred_y1, target_y1)
        enclosing_x2 = torch.max(pred_x2, target_x2)
        enclosing_y2 = torch.max(pred_y2, target_y2)
        enclosing_diagonal = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2

        # Compute aspect ratio consistency
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1
        target_w = target_x2 - target_x1
        target_h = target_y2 - target_y1

        v = 4 / (torch.pi ** 2) * ((torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2)

        # Compute cIoU
        ciou = iou - (center_dist / (enclosing_diagonal + 1e-7)) - v
        return ciou
