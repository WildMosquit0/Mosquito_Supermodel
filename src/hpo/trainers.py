from src.hpo.utils import HPOParameterSpace, Logger, CallbackManager
from ultralytics import YOLO
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import torch


class CustomDataset(Dataset):
    def __init__(self, image_paths: List[str], target_size: Tuple[int, int]):
        self.image_paths = image_paths
        self.target_size = target_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def resize_and_random_crop(self, image, annotations):
        target_height, target_width = self.target_size
        original_width, original_height = image.width, image.height
        if original_height < target_height or original_width < target_width:
            scale_factor = max(target_height / original_height, target_width / original_width)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            image = image.resize((new_width, new_height), Image.BILINEAR)
            annotations[:, 1:] *= torch.tensor([scale_factor, scale_factor, scale_factor, scale_factor])
        if image.height > target_height and image.width > target_width:
            top = random.randint(0, image.height - target_height)
            left = random.randint(0, image.width - target_width)
            image = image.crop((left, top, left + target_width, top + target_height))
            annotations[:, 1] = (annotations[:, 1] * original_width - left) / target_width
            annotations[:, 2] = (annotations[:, 2] * original_height - top) / target_height
            annotations[:, 3] /= target_width / original_width
            annotations[:, 4] /= target_height / original_height
        annotations[:, 1:].clamp_(0, 1)
        return image, annotations

    def load_annotations(self, annotation_path: str) -> torch.Tensor:
        annotations = []
        with open(annotation_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                label = int(parts[0])
                bbox = list(map(float, parts[1:]))
                annotations.append([label] + bbox)
        return torch.tensor(annotations)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        annotation_path = (
            image_path.replace("images", "labels")
            .replace(".jpg", ".txt")
            .replace(".png", ".txt")
        )
        if not os.path.exists(annotation_path):
            annotations = torch.empty((0, 5))
        else:
            annotations = self.load_annotations(annotation_path)
        image, annotations = self.resize_and_random_crop(image, annotations)
        transformed_image = self.transform(image)
        return transformed_image, annotations

    def __len__(self):
        return len(self.image_paths)


class StandardTrainer:
    def __init__(self, model: YOLO, data_config: Dict[str, Any], logger: Logger):
        self.model = model
        self.data_config = data_config
        self.logger = logger
        self.project = "runs/train"
        self.experiment_name = "experiment"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        training_params = self.data_config.get("training", {})
        output_params = self.data_config.get("output", {})
        data = self.data_config.get("data", {})
        dataset_path = "/home/bohbot/ultralytics/datasets/mos/all_mos_new/images/train"
        image_paths = [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        dataset = CustomDataset(
            image_paths=image_paths,
            target_size=(training_params["imgsz"], training_params["imgsz"]),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=training_params["batch"],
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
        self.model.train(
            data=data,
            epochs=training_params["epochs"],
            batch=training_params["batch"],
            imgsz=training_params["imgsz"],
            optimizer=training_params["optimizer"],
            lr0=training_params["lr0"],
            save_dir=output_params["output_dir"],
            project=self.project,
            name=self.experiment_name,
            device=str(self.device),
        )
