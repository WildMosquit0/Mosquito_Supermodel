import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageStat
from tqdm import tqdm
from itertools import islice

class AnnotationAnalyzer:
    def __init__(self, images_folder: str, annotations_folder: str, output_folder: str) -> None:
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.metadata = pd.DataFrame()

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image to 640x640 if it exceeds this size."""
        width, height = image.size
        if width > 640 or height > 640:
            image = image.resize((640, 640), Image.Resampling.LANCZOS)
        return image
    
    def _process_single_image(self, image_name: str) -> dict:
        image_path = os.path.join(self.images_folder, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        # Resize if needed
        original_width, original_height = image.size
        image = self._resize_if_needed(image)
        width, height = image.size  # Updated dimensions

        # Compute global image features
        brightness = self._calculate_brightness(image)
        contrast = self._calculate_contrast(image)
        entropy = self._calculate_entropy(image)

        # Read YOLO annotations corresponding to this image
        annotation_path = os.path.join(self.annotations_folder, os.path.splitext(image_name)[0] + ".txt")
        annotations = []
        if os.path.exists(annotation_path):
            annotations = self._read_yolo_annotations(annotation_path, original_width, original_height, width, height)

        num_objects = len(annotations)
        ann_areas = [ann["area"] for ann in annotations] if annotations else []
        avg_ann_area = sum(ann_areas) / len(ann_areas) if ann_areas else 0

        return {
            "Image": image_name,
            "Width": width,
            "Height": height,
            "Brightness": brightness,
            "Contrast": contrast,
            "Entropy": entropy,
            "Num_Objects": num_objects,
            "Avg_Ann_Area": avg_ann_area,
        }

    def _read_yolo_annotations(self, annotation_path: str, orig_w: int, orig_h: int, new_w: int, new_h: int) -> list:
        """Adjust bounding boxes if image was resized."""
        annotations = []
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        try:
            with open(annotation_path, "r") as file:
                for line in file.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x_center, y_center, w, h = map(float, parts)
                    x_center *= scale_x
                    y_center *= scale_y
                    w *= scale_x
                    h *= scale_y
                    area = (w * h) / (new_w * new_h)
                    annotations.append({"class": cls, "x_center": x_center, "y_center": y_center, "width": w, "height": h, "area": area})
        except Exception as e:
            print(f"Error reading annotation file {annotation_path}: {e}")
        return annotations

    def _calculate_brightness(self, image: Image.Image) -> float:
        return ImageStat.Stat(image.convert("L")).mean[0]

    def _calculate_contrast(self, image: Image.Image) -> float:
        return ImageStat.Stat(image.convert("L")).stddev[0]

    def _calculate_entropy(self, image: Image.Image) -> float:
        histogram = image.convert("L").histogram()
        total = sum(histogram)
        return -sum((count / total) * math.log2(count / total) for count in histogram if count != 0)

if __name__ == "__main__":
    images_folder = "/home/bohbot/workspace/datasets/mos/crop_without_bg/images/train"
    annotations_folder = "/home/bohbot/workspace/datasets/mos/crop_without_bg/labels/train"
    output_folder = "./output_annotation_analysis_crop"

    analyzer = AnnotationAnalyzer(images_folder, annotations_folder, output_folder)
    
    # Process all images
    metadata_list = []
    for image_name in tqdm(os.listdir(images_folder), desc="Processing Images"):
        if image_name.lower().endswith((".jpg", ".png")):
            meta = analyzer._process_single_image(image_name)
            if meta:
                metadata_list.append(meta)

    analyzer.metadata = pd.DataFrame(metadata_list)
    
    print("Feature extraction completed.")
