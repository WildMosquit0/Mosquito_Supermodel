from src.config.utils import ConfigLoader
from src.hpo.utils import Logger, CallbackManager
from src.hpo.trainers import StandardTrainer

from ultralytics import YOLO
from typing import Dict, List

class YOLOTrainer:
    def __init__(self, config_loader, use_hpo: bool = False):
        self.config_loader = config_loader
        self.use_hpo = use_hpo
        self.data_config = config_loader.get_data_config()

        self.config = self.config_loader.get_data_config()
        weights_path = self.config['model'].get('weights', 'yolov5s.pt')
        self.model = YOLO(weights_path)

        self.logger = Logger(log_dir='runs/train/experiment')
        self.callback_manager = CallbackManager()

    def train(self):
        trainer = StandardTrainer(
            model=self.model,
            data_config=self.data_config,
            logger=self.logger

        )
        trainer.train()


if __name__ == '__main__':
    config_loader = ConfigLoader('data.yaml')
    trainer = YOLOTrainer(config_loader)
    trainer.train()