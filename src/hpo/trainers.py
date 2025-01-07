from src.hpo.utils import HPOParameterSpace, Logger, CallbackManager
from ultralytics import YOLO
from typing import Dict, Any
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import torch


class CustomDataset:
    def __init__(self, image_paths, target_size):
        self.image_paths = image_paths
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Lambda(self.resize_and_random_crop),
            transforms.ToTensor()
        ])

    def resize_and_random_crop(self, image):
        target_height, target_width = self.target_size
        if image.height < target_height or image.width < target_width:
            scale_factor = max(target_height / image.height, target_width / image.width)
            new_height = int(image.height * scale_factor)
            new_width = int(image.width * scale_factor)
            image = image.resize((new_width, new_height), Image.BILINEAR)
        if image.height > target_height and image.width > target_width:
            top = random.randint(0, image.height - target_height)
            left = random.randint(0, image.width - target_width)
            image = image.crop((left, top, left + target_width, top + target_height))
        elif image.height != target_height or image.width != target_width:
            image = image.resize((target_width, target_height), Image.BILINEAR)
        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)


class StandardTrainer:
    def __init__(self, model: YOLO, data_config: Dict[str, Any], logger: Logger):
        self.model = model
        self.data_config = data_config
        self.logger = logger
        self.project = 'runs/train'
        self.experiment_name = 'experiment'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        training_params = self.data_config.get('training', {})
        output_params = self.data_config.get('output', {})
        data = self.data_config.get('data', {})

        dataset_path = '/home/bohbot/ultralytics/datasets/mos/all_mos_new/images/train'
        image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        dataset = CustomDataset(image_paths=image_paths, target_size=(training_params['imgsz'], training_params['imgsz']))
        dataloader = DataLoader(dataset, batch_size=training_params['batch'], shuffle=True, pin_memory=True, num_workers=4)

        self.model.train(
            data=data,
            epochs=training_params['epochs'],
            batch=training_params['batch'],
            imgsz=training_params['imgsz'],
            optimizer=training_params['optimizer'],
            lr0=training_params['lr0'],
            save_dir=output_params['output_dir'],
            project=self.project,
            name=self.experiment_name,
            device=str(self.device)
        )
