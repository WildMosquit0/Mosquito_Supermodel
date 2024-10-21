import os
from torch.utils.data import Dataset
from PIL import Image
import torch

from src.logger.logger import logger

class YOLODataset(Dataset):
    """
    Custom dataset for object detection with YOLO format annotations.
    
    The dataset supports two formats:
    1. A directory of images and corresponding YOLO .txt annotation files.
    2. A .txt file where each line contains the full path to an image and its corresponding annotation file.
    """
    def __init__(self, input_path, transform=None):
        """
        Args:
            input_path (str): Path to the folder containing images and .txt files, 
                              or a .txt file containing image and annotation paths.
            transform (callable, optional): Transform to apply to the images.
        """
        self.transform = transform
        self.image_paths = []
        self.annotation_paths = []

        if os.path.isdir(input_path):
            # Case 1: Image directory with corresponding YOLO .txt annotation files
            for fname in os.listdir(input_path):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(input_path, fname)
                    txt_path = os.path.join(input_path, fname.rsplit('.', 1)[0] + ".txt")
                    if os.path.exists(txt_path):
                        self.image_paths.append(img_path)
                        self.annotation_paths.append(txt_path)
                    else:
                        logger.info(f"Annotation file not found for image {img_path}")

        elif os.path.isfile(input_path) and input_path.endswith('.txt'):
            # Case 2: .txt file with image and annotation paths
            with open(input_path, 'r') as file:
                for line in file:
                    img_path, txt_path = line.strip().split(',')
                    if os.path.exists(img_path) and os.path.exists(txt_path):
                        self.image_paths.append(img_path)
                        self.annotation_paths.append(txt_path)
                    else:
                        raise FileNotFoundError(f"File not found: {img_path} or {txt_path}")

        else:
            raise ValueError("Input must be a directory of images or a .txt file with image and annotation paths.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Load YOLO-format annotation
        txt_path = self.annotation_paths[idx]
        with open(txt_path, 'r') as file:
            yolo_annotations = [list(map(float, line.strip().split())) for line in file]

        # Convert YOLO annotations from normalized format to absolute coordinates (if needed)
        yolo_annotations = torch.tensor(yolo_annotations)

        if self.transform:
            image = self.transform(image)

        return image, yolo_annotations
