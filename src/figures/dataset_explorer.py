import os
import base64

import pandas as pd
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from typing import List
from colorthief import ColorThief
import math
from concurrent.futures import ThreadPoolExecutor

import plotly.express as px
import umap
import os
from tqdm import tqdm
from itertools import islice

import torch
from torchvision import transforms

class ImageSetupAnalyzer:
    def __init__(self, root_folder: str, output_folder: str, thumbnail_size: tuple = (100, 100)) -> None:
        self.root_folder = root_folder
        self.output_folder = output_folder
        self.metadata: pd.DataFrame = pd.DataFrame()
        self.thumbnail_size = thumbnail_size
        os.makedirs(self.output_folder, exist_ok=True)

    def extract_metadata(self, batch_size: int = 200) -> None:
        metadata_list = []
        image_files = [
            os.path.join(self.root_folder, img)
            for img in os.listdir(self.root_folder)
            if img.lower().endswith(".jpg")
        ]

        def batch_iterator(iterable, size):
            iterator = iter(iterable)
            for first in iterator:
                yield [first] + list(islice(iterator, size - 1))

        batches = batch_iterator(image_files, batch_size)
        total_batches = len(image_files) // batch_size + (len(image_files) % batch_size > 0)

        for batch in tqdm(batches, total=total_batches, desc="Processing Batches on GPU"):
            metadata_list.extend(self._process_image_batch(batch))

        self.metadata = pd.DataFrame(metadata_list)

    def _process_image_batch(self, image_paths: List[str]) -> List[dict]:
        batch_images = []
        original_dimensions = []  # Store original dimensions for aspect ratio
        metadata_list = []

        # Resize transform
        resize_transform = transforms.Compose([
            transforms.Resize(self.thumbnail_size),  # Resize to thumbnail size
            transforms.ToTensor()  # Convert to tensor
        ])

        # Load images into tensors and store original dimensions
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                original_dimensions.append(image.size)  # Store (width, height)
                image_tensor = resize_transform(image).unsqueeze(0)  # Add batch dimension
                batch_images.append(image_tensor)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue

        if not batch_images:
            return metadata_list

        # Combine images into a single batch tensor
        batch_tensor = torch.cat(batch_images).to("cuda")  # Send to GPU

        # Calculate brightness and contrast
        brightness = batch_tensor.mean(dim=[1, 2, 3]).cpu().numpy() * 255
        contrast = batch_tensor.std(dim=[1, 2, 3]).cpu().numpy() * 255

        # Calculate aspect ratio from original dimensions
        aspect_ratios = [width / height for width, height in original_dimensions]

        # Grayscale and entropy calculation
        grayscale_batch = batch_tensor.mean(dim=1, keepdim=True).flatten(1)  # Flatten each image
        histograms = torch.stack([torch.histc(img, bins=256, min=0, max=1) for img in grayscale_batch], dim=0)
        histograms = histograms / histograms.sum(dim=1, keepdim=True)  # Normalize per image
        entropy = (-histograms * torch.log2(histograms + 1e-7)).sum(dim=1).cpu().numpy()  # Per image entropy

        # Generate metadata for each image
        for i, image_path in enumerate(image_paths):
            metadata_list.append({
                "Image": os.path.basename(image_path),
                "Width": original_dimensions[i][0],
                "Height": original_dimensions[i][1],
                "Brightness": brightness[i],
                "Contrast": contrast[i],
                "Aspect_Ratio": aspect_ratios[i],
                "Entropy": entropy[i],
            })

        return metadata_list




    def _process_image(self, image_path: str) -> dict:
        try:
            # Open and validate the image
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error opening {image_path}: {e}")
                return None

            if image.size == (0, 0):
                print(f"Invalid image size for {image_path}")
                return None

            # Convert to tensor
            image_tensor = transforms.ToTensor()(image).to("cuda")  # Send to GPU

            # Calculate brightness and contrast
            brightness = image_tensor.mean().item() * 255
            contrast = image_tensor.std().item() * 255

            # Resize for aspect ratio calculation
            image_resized = transforms.Resize(self.thumbnail_size)(image)
            aspect_ratio = image_resized.size[0] / image_resized.size[1]

            # Grayscale and entropy (back to CPU)
            image_grayscale = transforms.Grayscale()(image_resized)
            image_grayscale_tensor = transforms.ToTensor()(image_grayscale).to("cuda").flatten()

            histogram = torch.histc(image_grayscale_tensor, bins=256, min=0, max=1)  # Adjust range for normalized grayscale
            histogram = histogram / histogram.sum()
            entropy = -torch.sum(histogram * torch.log2(histogram + 1e-7)).item()


            # Dominant color (still on CPU)
            dominant_color = self._extract_dominant_color(image_path)

            return {
                "Image": os.path.basename(image_path),
                "Width": image.size[0],
                "Height": image.size[1],
                "Brightness": brightness,
                "Contrast": contrast,
                "Aspect_Ratio": aspect_ratio,
                "Dominant_Color_R": dominant_color[0],
                "Dominant_Color_G": dominant_color[1],
                "Dominant_Color_B": dominant_color[2],
                "Entropy": entropy,
            }
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
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
        return -sum((count / total) * (math.log2(count / total)) for count in histogram if count > 0)

    def _extract_dominant_color(self, image_path: str) -> List[int]:
        color_thief = ColorThief(image_path)
        return color_thief.get_color(quality=1)

    def save_feature_distributions(self, features: List[str], output_folder: str) -> None:
        # Check if the output path exists
        if os.path.exists(output_folder):
            if not os.path.isdir(output_folder):
                # If it exists and is not a directory, delete it
                os.remove(output_folder)
                os.makedirs(output_folder)
        else:
            os.makedirs(output_folder)

        if self.metadata.empty:
            print("No metadata to plot. Run extract_metadata() first.")
            return

        for feature in features:
            if feature not in self.metadata.columns:
                print(f"Feature {feature} is not available in the metadata.")
                continue
            plt.figure()
            sns.histplot(data=self.metadata, x=feature, kde=True)
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            output_path = os.path.join(output_folder, f"{feature}_distribution.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Saved: {output_path}")



    def save_correlation_matrix(self, output_folder: str) -> None:
        if self.metadata.empty:
            print("No metadata to analyze. Run extract_metadata() first.")
            return

        os.makedirs(output_folder, exist_ok=True)
        corr_matrix = self.metadata.drop(columns=["Image"]).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlation Matrix")
        output_path = os.path.join(output_folder, "correlation_matrix.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def save_pca_plot(self, n_components: int, output_folder: str) -> None:
        if self.metadata.empty:
            print("No metadata to analyze. Run extract_metadata() first.")
            return

        os.makedirs(output_folder, exist_ok=True)
        features = self.metadata.drop(columns=["Image"])
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(features)
        pca_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(n_components)])
        plt.figure()
        sns.scatterplot(data=pca_df, x="PC1", y="PC2")
        plt.title("PCA of Features")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        output_path = os.path.join(output_folder, "pca_plot.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")


    def plot_umap_pca_html(self, method: str = "UMAP", n_components: int = 2, output_file: str = "output.html") -> None:
        if self.metadata.empty:
            print("No metadata to analyze. Run extract_metadata() first.")
            return

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Select features for dimensionality reduction
        features = self.metadata[["Brightness", "Contrast", "Aspect_Ratio", "Entropy"]].values
        image_paths = self.metadata["Image"].values
        image_full_paths = [os.path.join(self.root_folder, img) for img in image_paths]

        # Encode images in Base64
        def encode_image_to_base64(image_path):
            try:
                with open(image_path, "rb") as image_file:
                    return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}"
            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")
                return None

        image_base64 = [encode_image_to_base64(img) for img in image_full_paths]

        # Perform dimensionality reduction
        if method.upper() == "UMAP":
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            embeddings = reducer.fit_transform(features)
        elif method.upper() == "PCA":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
            embeddings = reducer.fit_transform(features)
        else:
            print(f"Invalid method '{method}'. Choose 'UMAP' or 'PCA'.")
            return

        # Prepare data for Plotly
        data = {
            "X": embeddings[:, 0],
            "Y": embeddings[:, 1],
            "Image_Base64": image_base64,
            "Brightness": self.metadata["Brightness"],
            "Contrast": self.metadata["Contrast"],
            "Aspect_Ratio": self.metadata["Aspect_Ratio"],
            "Entropy": self.metadata["Entropy"]
        }

        # Create a Plotly scatter plot
        fig = px.scatter(
            data,
            x="X",
            y="Y",
            title=f"{method} Visualization of Image Features",
            hover_data=["Brightness", "Contrast", "Aspect_Ratio", "Entropy"],
            custom_data=["Image_Base64"]
        )

        # Add hoverable thumbnails
        fig.update_traces(
            marker=dict(size=8, opacity=0.8),
            hovertemplate="<b>Image:</b><br>" +
                        "<img src='%{customdata[0]}' style='max-width:80px; max-height:80px;'><br>" +
                        "<b>X:</b> %{x}<br>" +
                        "<b>Y:</b> %{y}<br>" +
                        "<b>Brightness:</b> %{hoverdata[0]}<br>" +
                        "<b>Contrast:</b> %{hoverdata[1]}<br>" +
                        "<b>Aspect Ratio:</b> %{hoverdata[2]}<br>" +
                        "<b>Entropy:</b> %{hoverdata[3]}"
        )

        # Save to an HTML file
        outpath = f"{output_file}/vis.html"
        fig.write_html(outpath)
        print(f"Interactive plot saved to {outpath}")


if __name__ == "__main__":
    folder_path = "/home/bohbot/workspace/datasets/mos/all_mos_new/images/train"
    output_folder = "./output_images"

    analyzer = ImageSetupAnalyzer(folder_path, output_folder)
    analyzer.extract_metadata()
    
    analyzer.plot_umap_pca_html(
        method="PCA",               # Choose "UMAP" or "PCA"
        n_components=2,              # Number of dimensions
        output_file=output_folder      # Output HTML file
    )

    features_to_plot = ["Brightness", "Contrast", "Aspect_Ratio"]
    analyzer.save_feature_distributions(features=features_to_plot, output_folder=output_folder)
    analyzer.save_correlation_matrix(output_folder=output_folder)
    analyzer.save_pca_plot(n_components=2, output_folder=output_folder)
