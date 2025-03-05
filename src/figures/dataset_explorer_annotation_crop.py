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

    def extract_metadata(self, batch_size: int = 200) -> None:
        """This method is not used in the current workflow since filtering is done in main."""
        metadata_list = []
        image_files = [img for img in os.listdir(self.images_folder) if img.lower().endswith((".jpg", ".png"))]
        image_files = sorted(image_files)

        def batch_iterator(iterable, size):
            iterator = iter(iterable)
            for first in iterator:
                yield [first] + list(islice(iterator, size - 1))

        batches = batch_iterator(image_files, batch_size)
        total_batches = len(image_files) // batch_size + (len(image_files) % batch_size > 0)

        for batch in tqdm(batches, total=total_batches, desc="Processing Image Batches"):
            for image_name in batch:
                meta = self._process_single_image(image_name)
                if meta:
                    metadata_list.append(meta)

        self.metadata = pd.DataFrame(metadata_list)

    def _process_single_image(self, image_name: str) -> dict:
        image_path = os.path.join(self.images_folder, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        width, height = image.size
        image_area = width * height

        # Compute global image features
        brightness = self._calculate_brightness(image)
        contrast = self._calculate_contrast(image)
        entropy = self._calculate_entropy(image)

        # Read YOLO annotations corresponding to this image
        annotation_path = os.path.join(self.annotations_folder, os.path.splitext(image_name)[0] + ".txt")
        annotations = []
        if os.path.exists(annotation_path):
            annotations = self._read_yolo_annotations(annotation_path, width, height)

        num_objects = len(annotations)
        # Calculate annotation areas, widths, and heights
        ann_areas = [ann["area"] for ann in annotations] if annotations else []
        bb_widths = [ann["width"] for ann in annotations] if annotations else []
        bb_heights = [ann["height"] for ann in annotations] if annotations else []

        avg_ann_area = sum(ann_areas) / len(ann_areas) if ann_areas else 0
        min_ann_area = min(ann_areas) if ann_areas else 0
        max_ann_area = max(ann_areas) if ann_areas else 0
        
        avg_bb_width = sum(bb_widths) / len(bb_widths) if bb_widths else 0
        min_bb_width = min(bb_widths) if bb_widths else 0
        max_bb_width = max(bb_widths) if bb_widths else 0
        
        avg_bb_height = sum(bb_heights) / len(bb_heights) if bb_heights else 0
        min_bb_height = min(bb_heights) if bb_heights else 0
        max_bb_height = max(bb_heights) if bb_heights else 0

        return {
            "Image": image_name,
            "Width": width,
            "Height": height,
            "Image_Area": image_area,
            "Brightness": brightness,
            "Contrast": contrast,
            "Entropy": entropy,
            "Num_Objects": num_objects,
            "Avg_Ann_Area": avg_ann_area,
            "Min_Ann_Area": min_ann_area,
            "Max_Ann_Area": max_ann_area,
            "Avg_BB_Width": avg_bb_width,
            "Min_BB_Width": min_bb_width,
            "Max_BB_Width": max_bb_width,
            "Avg_BB_Height": avg_bb_height,
            "Min_BB_Height": min_bb_height,
            "Max_BB_Height": max_bb_height,
        }

    def _read_yolo_annotations(self, annotation_path: str, img_width: int, img_height: int) -> list:
        """
        Reads a YOLO-format annotation file.
        Each line is expected to have: class x_center y_center width height (all normalized).
        Converts normalized values to pixel coordinates and computes the bounding box area.
        """
        annotations = []
        try:
            with open(annotation_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) != 5:
                            print(f"Invalid annotation format in {annotation_path}: {line}")
                            continue
                        cls, x_center, y_center, w, h = parts
                        x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
                        # Convert normalized values to pixel dimensions
                        box_width = w * img_width 
                        box_height = h * img_height
                        area = box_width * box_height / 409600
                        annotations.append({
                            "class": cls,
                            "x_center": x_center * img_width,
                            "y_center": y_center * img_height,
                            "width": box_width,
                            "height": box_height,
                            "area": area
                        })
        except Exception as e:
            print(f"Error reading annotation file {annotation_path}: {e}")
        return annotations

    def _calculate_brightness(self, image: Image.Image) -> float:
        grayscale = image.convert("L")
        stat = ImageStat.Stat(grayscale)
        return stat.mean[0]

    def _calculate_contrast(self, image: Image.Image) -> float:
        grayscale = image.convert("L")
        stat = ImageStat.Stat(grayscale)
        return stat.stddev[0]

    def _calculate_entropy(self, image: Image.Image) -> float:
        grayscale = image.convert("L")
        histogram = grayscale.histogram()
        total = sum(histogram)
        entropy = -sum((count / total) * math.log2(count / total) for count in histogram if count != 0)
        return entropy

    def save_feature_distributions(self, features: list) -> None:
        if self.metadata.empty:
            print("No metadata available. Run extract_metadata() first.")
            return
        for feature in features:
            if feature not in self.metadata.columns:
                print(f"Feature {feature} not found in metadata.")
                continue
            mean_val = self.metadata[feature].mean()
            max_val = self.metadata[feature].max()
            plt.figure()
            sns.histplot(self.metadata[feature], kde=True)
            plt.title(f"Distribution of {feature}\n(Mean: {mean_val:.2f}, Max: {max_val:.2f})")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            output_path = os.path.join(self.output_folder, f"{feature}_distribution.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Saved distribution plot: {output_path}")

    def save_correlation_matrix(self) -> None:
        if self.metadata.empty:
            print("No metadata available. Run extract_metadata() first.")
            return
        plt.figure(figsize=(10, 8))
        # Drop the image name column and min/max bbox columns from correlation calculation
        drop_cols = ["Min_Ann_Area", "Max_Ann_Area", "Min_BB_Width", "Max_BB_Width", "Min_BB_Height", "Max_BB_Height"]
        cols_to_drop = ["Image"] + [col for col in drop_cols if col in self.metadata.columns]
        corr_df = self.metadata.drop(columns=cols_to_drop)
        corr_matrix = corr_df.corr()

        # Compute off-diagonal mean and max correlation (excluding self-correlations which are 1)
        off_diag = corr_matrix.where(~(corr_matrix == 1)).stack()
        mean_corr = off_diag.mean()
        max_corr = off_diag.max()

        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Feature Correlation Matrix\n(Mean Corr: {mean_corr:.2f}, Max Corr: {max_corr:.2f})")
        output_path = os.path.join(self.output_folder, "correlation_matrix.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved correlation matrix plot: {output_path}")

    def save_scatter_plots(self) -> None:
        if self.metadata.empty:
            print("No metadata available. Run extract_metadata() first.")
            return

        # ----- Box Plot: Image Area by Number of Objects -----
        plt.figure()
        sns.boxplot(data=self.metadata, x="Num_Objects", y="Image_Area")
        mean_img_area = self.metadata["Image_Area"].mean()
        max_img_area = self.metadata["Image_Area"].max()
        max_objects = self.metadata["Num_Objects"].max()
        images_with_max = self.metadata[self.metadata["Num_Objects"] == max_objects]["Image"].tolist()
        max_images_str = ", ".join(images_with_max)
        plt.title(f"Image Area by Number of Objects\n(Max Objects: {max_objects} in {max_images_str}\nMean Img Area: {mean_img_area:.2f}, Max Img Area: {max_img_area:.2f})")
        plt.xlabel("Number of Objects")
        plt.ylabel("Image Area (pixels)")
        output_boxplot = os.path.join(self.output_folder, "num_objects_vs_image_area_boxplot.png")
        plt.savefig(output_boxplot)
        plt.close()
        print(f"Saved box plot (Num_Objects vs Image_Area): {output_boxplot}")
        print(f"Image(s) with maximum objects ({max_objects}): {max_images_str}")

        # ----- Scatter Plot: Average Annotation Area vs Image Area -----
        plt.figure()
        sns.scatterplot(data=self.metadata, x="Image_Area", y="Avg_Ann_Area")
        mean_avg_ann_area = self.metadata["Avg_Ann_Area"].mean()
        max_avg_ann_area = self.metadata["Avg_Ann_Area"].max()
        plt.title(f"Average Annotation Area vs Image Area\n(Mean Avg Ann Area: {mean_avg_ann_area:.2f}, Max Avg Ann Area: {max_avg_ann_area:.2f})")
        plt.xlabel("Image Area (pixels)")
        plt.ylabel("Average Annotation Area (pixels)")
        output_scatter = os.path.join(self.output_folder, "avg_annotation_area_vs_image_area.png")
        plt.savefig(output_scatter)
        plt.close()
        print(f"Saved scatter plot (Avg_Ann_Area vs Image_Area): {output_scatter}")

if __name__ == "__main__":
    # Set your own paths here
    images_folder = "/home/bohbot/workspace/datasets/mos/crop_without_bg/images/train"
    annotations_folder = "/home/bohbot/workspace/datasets/mos/crop_without_bg/labels/train"
    output_folder = "./output_annotation_analysis_crop" 

    analyzer = AnnotationAnalyzer(images_folder, annotations_folder, output_folder)
    
    # --- Filtering Images in Main Block Based on Dimensions ---
    # Only include images with dimensions ≤ 640×640.
    valid_images = []
    for img_name in os.listdir(images_folder):
        if img_name.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(images_folder, img_name)
            try:
                with Image.open(img_path) as im:
                    width, height = im.size
                    if width <= 640 and height <= 640:
                        valid_images.append(img_name)
                    else:
                        print(f"Skipping image {img_name} as its dimensions {width}x{height} exceed 640x640.")
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")

    # Process only the filtered images
    metadata_list = []
    for image_name in valid_images:
        meta = analyzer._process_single_image(image_name)
        if meta:
            metadata_list.append(meta)
    analyzer.metadata = pd.DataFrame(metadata_list)
    
    # Generate plots with computed statistics.
    features_to_plot = [
        "Brightness", "Contrast", "Entropy", 
        "Num_Objects", "Avg_Ann_Area", "Avg_BB_Width", "Avg_BB_Height"
    ]
    analyzer.save_feature_distributions(features_to_plot)
    analyzer.save_correlation_matrix()
    analyzer.save_scatter_plots()

    print("Annotation analysis completed.")
