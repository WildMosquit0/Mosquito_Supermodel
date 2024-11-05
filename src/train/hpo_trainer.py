from ultralytics import YOLO
from typing import Dict, Any
import optuna
import logging
from src.utils.hpo_utils import create_data_yaml
from src.logger.logger import Logger
from src.hpo.utils import HPOParameterSpace, CallbackManager

logging.basicConfig(level=logging.INFO)

class HPOTrainer:
    """Manages hyperparameter optimization with YOLO using Optuna."""

    def __init__(self, model: YOLO, data_config: Dict[str, Any], hyp_config: Dict[str, Any], 
                 hpo_space: HPOParameterSpace, logger: Logger, callback_manager: CallbackManager):
        self.model = model
        self.data_config = data_config
        self.hyp_config = hyp_config
        self.hpo_space = hpo_space
        self.logger = logger
        self.callback_manager = callback_manager
        self.output_dir = self.data_config.get('output', {}).get('output_dir', 'runs/hpo')
        self.experiment_name = 'experiment'

    def train(self) -> None:
        """Starts the Optuna optimization process."""
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.hyp_config.get("hpo_params", {}).get("iterations", 50))

    def _objective(self, trial: optuna.Trial) -> float:
        """Defines the objective function for Optuna to optimize."""
        try:
            # Suggest hyperparameters using Optuna based on hpo.yaml configuration
            lr0 = trial.suggest_float("lr0", self.hyp_config["lr0"]["min"], self.hyp_config["lr0"]["max"], log=True)
            batch = trial.suggest_categorical("batch", self.hyp_config["batch"]["values"])
            optimizer = trial.suggest_categorical("optimizer", self.hyp_config["hpo_params"]["optimizer"]["values"])
            momentum = trial.suggest_float("momentum", self.hyp_config["momentum"]["min"], self.hyp_config["momentum"]["max"])
            weight_decay = trial.suggest_float("weight_decay", self.hyp_config["weight_decay"]["min"], self.hyp_config["weight_decay"]["max"])
            epochs = self.hyp_config["hpo_params"]["epochs"]

            # Prepare data YAML file
            data_yaml = create_data_yaml(self.hyp_config['data'])

            # Run training with the chosen hyperparameters
            results = self.model.train(
                data=data_yaml,
                imgsz=self.data_config['training']['imgsz'],
                epochs=epochs,
                batch=batch,
                lr0=lr0,
                optimizer=optimizer,
                momentum=momentum,
                weight_decay=weight_decay,
                project=self.output_dir,
                name=self.experiment_name,
                save=True
            )

            # Extract fitness and other metrics from results
            fitness = results.fitness
            precision = results.box.map50  # Replace with actual attribute if different
            recall = results.box.map50_95  # Replace with actual attribute if different

            # Log metrics for analysis
            metrics_dict = {
                "fitness": fitness,
                "precision": precision,
                "recall": recall,
            }
            self.logger.log_metrics(trial.number, metrics_dict)

            # Return the primary metric used for optimization
            return fitness

        except Exception as e:
            logging.error(f"An error occurred during Optuna optimization: {e}")
            raise

