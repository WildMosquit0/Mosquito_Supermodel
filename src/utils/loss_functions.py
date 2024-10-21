import torch
from src.utils.box_utils import compute_box_iou, compute_center_distances, compute_enclosing_box, compute_aspect_ratio_loss

def compute_ciou_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_boxes = predictions[..., :4]
    target_boxes = targets[..., :4]
    
    iou = compute_box_iou(pred_boxes, target_boxes)
    center_dist = compute_center_distances(pred_boxes, target_boxes)
    enclosing_diagonal = compute_enclosing_box(pred_boxes, target_boxes)
    v = compute_aspect_ratio_loss(pred_boxes, target_boxes)
    
    ciou_loss = 1 - iou + (center_dist / (enclosing_diagonal + 1e-7)) + v
    return ciou_loss.mean()
