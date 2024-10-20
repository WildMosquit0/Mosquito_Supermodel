import torch
from torchvision.ops import box_iou

def compute_box_iou(pred_boxes, target_boxes):
    """
    Compute IoU between predicted and target bounding boxes using torchvision's optimized function.
    
    Args:
        pred_boxes (Tensor): Predicted bounding boxes [x1, y1, x2, y2]
        target_boxes (Tensor): Target (ground truth) bounding boxes [x1, y1, x2, y2]
    
    Returns:
        Tensor: IoU matrix between each predicted and target box.
    """
    return box_iou(pred_boxes, target_boxes)

def compute_enclosing_box(pred_boxes, target_boxes):
    """
    Compute the smallest enclosing box for predicted and target boxes, used for cIoU.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes [x1, y1, x2, y2]
        target_boxes (Tensor): Target (ground truth) bounding boxes [x1, y1, x2, y2]
    
    Returns:
        Tensor: Enclosing box diagonal distances.
    """
    enclosing_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclosing_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclosing_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclosing_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    enclosing_diagonal = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2
    return enclosing_diagonal

def compute_center_distances(pred_boxes, target_boxes):
    """
    Compute the squared distances between the centers of the predicted and target boxes.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes [x1, y1, x2, y2]
        target_boxes (Tensor): Target (ground truth) bounding boxes [x1, y1, x2, y2]
    
    Returns:
        Tensor: Squared distances between box centers.
    """
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    center_dist = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
    return center_dist

def compute_aspect_ratio_loss(pred_boxes, target_boxes):
    """
    Compute aspect ratio consistency loss for cIoU.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes [x1, y1, x2, y2]
        target_boxes (Tensor): Target (ground truth) bounding boxes [x1, y1, x2, y2]
    
    Returns:
        Tensor: Aspect ratio loss (v) term.
    """
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]
    
    v = (4 / (torch.pi ** 2)) * ((torch.atan(target_w / target_h + 1e-7) - torch.atan(pred_w / pred_h + 1e-7)) ** 2)
    return v
