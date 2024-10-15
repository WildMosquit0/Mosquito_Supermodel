import torch
import torch.nn as nn

def bbox_iou(box1, box2, eps=1e-7):
    """Calculate IoU (Intersection over Union) between two bounding boxes."""
    # Get the coordinates of bounding boxes
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter_area_x1 = torch.max(box1_x1, box2_x1)
    inter_area_y1 = torch.max(box1_y1, box2_y1)
    inter_area_x2 = torch.min(box1_x2, box2_x2)
    inter_area_y2 = torch.min(box1_y2, box2_y2)
    inter_area = torch.clamp(inter_area_x2 - inter_area_x1, min=0) * torch.clamp(inter_area_y2 - inter_area_y1, min=0)

    # Union Area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area + eps

    return inter_area / union_area

def compute_ciou_loss(predictions, targets):
    """
    Compute the Complete IoU (cIOU) loss between predictions and targets.
    Args:
        predictions: Predicted bounding boxes [x1, y1, x2, y2] (Tensor)
        targets: Ground-truth bounding boxes [x1, y1, x2, y2] (Tensor)
    
    Returns:
        torch.Tensor: cIOU loss value
    """
    # Extract predicted and target boxes
    pred_boxes = predictions[..., :4]  # YOLO typically predicts box as the first 4 elements
    target_boxes = targets[..., :4]    # Assuming the same format for targets

    # Compute IoU
    iou = bbox_iou(pred_boxes, target_boxes)

    # Calculate the center distances
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    center_dist = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2

    # Enclosing box (smallest box that encloses both the prediction and the target)
    enclosing_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclosing_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclosing_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclosing_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    enclosing_diagonal = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2

    # Aspect ratio consistency
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]
    aspect_ratio_loss = 4 / (torch.pi ** 2) * ((torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2)

    # cIOU Loss calculation
    ciou_loss = 1 - iou + (center_dist / (enclosing_diagonal + 1e-7)) + aspect_ratio_loss
    return ciou_loss.mean()

