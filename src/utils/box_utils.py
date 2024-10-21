import torch
from torchvision.ops import box_iou

def compute_box_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    return box_iou(pred_boxes, target_boxes)

def compute_enclosing_box(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    enclosing_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclosing_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclosing_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclosing_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    return (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2

def compute_center_distances(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    return (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2

def compute_aspect_ratio_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]
    return (4 / (torch.pi ** 2)) * ((torch.atan(target_w / target_h + 1e-7) - torch.atan(pred_w / pred_h + 1e-7)) ** 2)
