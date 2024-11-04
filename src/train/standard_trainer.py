from ultralytics import YOLO
from typing import Dict, Any
import logging
from src.logger.logger import Logger

logging.basicConfig(level=logging.INFO)

class StandardTrainer:
    """Handles standard training with YOLO and logs results."""

    def __init__(self, model: YOLO, data_config: Dict[str, Any], hyp_config: Dict[str, Any], logger: Logger):
        self.model = model
        self.data_config = data_config
        self.hyp_config = hyp_config
        self.logger = logger
        self.output_dir = self.data_config.get('output', {}).get('output_dir', 'runs/train')
        self.experiment_name = 'experiment'

    def train(self) -> None:
        try:
            training_params = self.data_config.get('training', {})
            lr0 = self.hyp_config.get('lr0', 0.001)

            logging.info("Starting standard training with parameters:")
            logging.info(f"Image Size: {training_params['imgsz']}, Epochs: {training_params['epochs']}")

            results = self.model.train(
                data=self.data_config['path'],
                epochs=training_params['epochs'],
                batch=training_params['batch'],
                imgsz=training_params['imgsz'],
                optimizer=training_params['optimizer'],
                lr0=lr0,
                save_dir=self.output_dir,
                project=self.output_dir,
                name=self.experiment_name
            )

            metrics = results.metrics
            metrics_dict = {'fitness': metrics.fitness, 'precision': metrics.box.map50, 'recall': metrics.box.map50_95}
            hparams_dict = {'lr0': lr0, 'batch': training_params['batch'], 'epochs': training_params['epochs']}
            self.logger.log_hparams(hparams_dict, metrics_dict)
            self.logger.close()
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise
