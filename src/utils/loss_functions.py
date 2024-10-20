import torch
from src.utils.box_utils import compute_box_iou, compute_center_distances, compute_enclosing_box, compute_aspect_ratio_loss

def compute_ciou_loss(predictions, targets):
    """
    Compute the Complete IoU (cIoU) loss between predictions and targets.
    
    Args:
        predictions: Predicted bounding boxes [x1, y1, x2, y2] (Tensor)
        targets: Ground-truth bounding boxes [x1, y1, x2, y2] (Tensor)
    
    Returns:
        torch.Tensor: cIoU loss value
    """
    pred_boxes = predictions[..., :4]  # YOLO typically predicts box as the first 4 elements
    target_boxes = targets[..., :4]    # Assuming the same format for targets

    # IoU computation
    iou = compute_box_iou(pred_boxes, target_boxes)

    # Center distance and enclosing box
    center_dist = compute_center_distances(pred_boxes, target_boxes)
    enclosing_diagonal = compute_enclosing_box(pred_boxes, target_boxes)

    # Aspect ratio consistency
    v = compute_aspect_ratio_loss(pred_boxes, target_boxes)

    # cIoU: IoU - center distance / diagonal - aspect ratio consistency
    ciou_loss = 1 - iou + (center_dist / (enclosing_diagonal + 1e-7)) + v
    return ciou_loss.mean()
