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
            data=self.data_config['path'],
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


class HPOTrainer:
    def __init__(self, model: YOLO, data_config: Dict[str, Any], hyp_config: Dict[str, Any], 
                 hpo_space: HPOParameterSpace, logger: Logger, callback_manager: CallbackManager):
        self.model = model
        self.data_config = data_config
        self.hyp_config = hyp_config
        self.hpo_space = hpo_space
        self.logger = logger
        self.callback_manager = callback_manager
        self.project = 'runs/hpo'
        self.experiment_name = 'experiment'
        self.log_dir = os.path.join(self.project, self.experiment_name)
        self.iteration_counter = 0

    def train(self):
        space = self.hpo_space.get_space()
        
        # Ensure all values in `space` are correctly typed (floats/ints)
        def parse_space_value(value):
            if isinstance(value, list):
                return [float(v) if isinstance(v, str) and v.replace('.', '', 1).isdigit() else v for v in value]
            return float(value) if isinstance(value, str) and value.replace('.', '', 1).isdigit() else value

        space = {k: parse_space_value(v) for k, v in space.items()}

        data_yaml = self._create_data_yaml(self.hyp_config['data'])

        print("Model overrides before tuning:", self.model.overrides)

        batch = self.hyp_config.get('batch', {}).get('values', [16])[0]  # Use `batch` instead of `batch_size`

        print("Starting HPO with space:", space)

        def on_train_start(trainer):
            self.iteration_counter += 1
            iteration_log_dir = os.path.join(self.log_dir, f'iteration_{self.iteration_counter}')
            trainer.writer = SummaryWriter(log_dir=iteration_log_dir)
            hparams = {k: v for k, v in trainer.hyp.items()}
            trainer.hparams = hparams

        def on_train_end(trainer):
            metrics = trainer.metrics
            metrics_dict = {'fitness': metrics.fitness, 'precision': metrics.box.map50, 'recall': metrics.box.map50_95}
            trainer.writer.add_hparams(trainer.hparams, metrics_dict)
            trainer.writer.close()

        self.callback_manager.add_callback("on_train_start", on_train_start)
        self.callback_manager.add_callback("on_train_end", on_train_end)

        self.model.tune(
            data=data_yaml,
            imgsz=self.data_config['training']['imgsz'],
            epochs=self.hyp_config['hpo_params']['epochs'],
            iterations=self.hyp_config['hpo_params']['iterations'],
            save=True,
            plots=False,
            val=True,
            project=self.project,
            name=self.experiment_name,
            space=space,
            batch=batch
        )

    def _create_data_yaml(self, data_dict):
        data_yaml_path = 'temp_data.yaml'
        with open(data_yaml_path, 'w') as file:
            yaml.dump(data_dict, file)
        return data_yaml_path

