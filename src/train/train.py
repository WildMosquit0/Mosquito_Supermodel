from src.config.utils import ConfigLoader
from src.hpo.utils import HPOParameterSpace, Logger, CallbackManager
from src.hpo.trainers import StandardTrainer, HPOTrainer

from ultralytics import YOLO
from typing import Dict, List

class YOLOTrainer:
    def __init__(self, config_loader, use_hpo: bool = False):
        self.config_loader = config_loader
        self.use_hpo = use_hpo
        self.data_config = config_loader.get_data_config()
        self.hyp_config = config_loader.get_hyp_config()

        weights_path = self.hyp_config.get('model', {}).get('weights', 'yolov8n.pt')
        self.model = YOLO(weights_path)

        self.logger = Logger(log_dir='runs/train/experiment')
        self.callback_manager = CallbackManager()

    def train(self, hpo_space=None):
        if self.use_hpo:
            trainer = HPOTrainer(
                model=self.model,
                data_config=self.data_config,
                hyp_config=self.hyp_config,
                hpo_space=hpo_space,
                logger=self.logger,
                callback_manager=self.callback_manager
            )
        else:
            trainer = StandardTrainer(
                model=self.model,
                data_config=self.data_config,
                hyp_config=self.hyp_config,
                logger=self.logger
            )

        trainer.train()


if __name__ == '__main__':
    config_loader = ConfigLoader('data.yaml', 'hpo.yaml')
    hpo_space = HPOParameterSpace(config_loader.get_hyp_config())
    trainer = YOLOTrainer(config_loader, use_hpo=True)
    trainer.train(hpo_space=hpo_space)