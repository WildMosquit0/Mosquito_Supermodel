from src.hpo.utils import HPOParameterSpace, Logger, CallbackManager
from ultralytics import YOLO
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import yaml
import torch


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
        data_params = self.data_config.get("data", {})
        temp_data_yaml_path = "/tmp/temp_data.yaml"
        with open(temp_data_yaml_path, "w") as f:
            yaml.dump(
                {
                    "train": os.path.join(data_params["path"], data_params["train"]),
                    "val": os.path.join(data_params["path"], data_params["val"]),
                    "test": os.path.join(
                        data_params["path"], data_params.get("test", "")
                    ),
                    "nc": data_params["nc"],
                    "names": data_params["names"],
                },
                f,
            )
        training_params["augmentations"] = self.add_random_crop(
            training_params["augmentations"], training_params["imgsz"]
        )
        augmentation_params = {
            k: v for k, v in training_params["augmentations"].items()
        }
        self.model.train(
            data=temp_data_yaml_path,
            epochs=training_params["epochs"],
            batch=training_params["batch"],
            imgsz=training_params["imgsz"],
            optimizer=training_params["optimizer"],
            lr0=training_params["lr0"],
            save_dir=output_params["output_dir"],
            project=self.project,
            name=self.experiment_name,
            device=str(self.device),
            **augmentation_params
        )

    def add_random_crop(self, transform_dict: dict, size: int):
        transform_dict['crop_fraction'] = (size, size)
        return transform_dict
