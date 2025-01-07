from src.hpo.utils import HPOParameterSpace, Logger, CallbackManager

from ultralytics import YOLO
from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter
import os
import yaml


class StandardTrainer:
    def __init__(self, model: YOLO, data_config: Dict[str, Any], hyp_config: Dict[str, Any], logger: Logger):
        self.model = model
        self.data_config = data_config
        self.hyp_config = hyp_config
        self.logger = logger
        self.project = 'runs/train'
        self.experiment_name = 'experiment'

    def train(self):
        training_params = self.data_config.get('training', {})
        output_params = self.data_config.get('output', {})
        lr0 = self.hyp_config.get('lr0', 0.001)

        print("Starting standard training with parameters:")
        print(f"Image Size: {training_params['imgsz']}, Epochs: {training_params['epochs']}")

        results = self.model.train(
            data='/home/bohbot/Evyatar/yaml/all_mos.yaml',
            epochs=training_params['epochs'],
            batch=training_params['batch'],
            imgsz=training_params['imgsz'],
            optimizer=training_params['optimizer'],
            lr0=lr0,
            save_dir=output_params['output_dir'],
            project=self.project,
            name=self.experiment_name
        )

        metrics = results.metrics
        metrics_dict = {'fitness': metrics.fitness, 'precision': metrics.box.map50, 'recall': metrics.box.map50_95}
        hparams_dict = {'lr0': lr0, 'batch': training_params['batch'], 'epochs': training_params['epochs']}
        self.logger.log_hparams(hparams_dict, metrics_dict)
        self.logger.close()
