import os
import logging
from ultralytics import YOLO
from src.train.hpo_trainer import HPOTrainer
from src.logger.logger import Logger
from src.hpo.utils import HPOParameterSpace, CallbackManager
from src.utils.hpo_utils import create_data_yaml

logging.basicConfig(level=logging.INFO)

def load_yaml_config(file_path: str):
    """Loads a YAML configuration file."""
    import yaml
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configurations
    data_config = load_yaml_config("data.yaml")
    hyp_config = load_yaml_config("hpo.yaml")
    model_config = load_yaml_config("hyp.yaml")  # Assuming model hyperparams are also in hyp.yaml

    # Initialize YOLO model
    model_weights = hyp_config["model"]["weights"]
    model = YOLO(model_weights)

    # Initialize logger, HPO space, and callback manager
    logger = Logger(log_dir="runs/hpo")
    hpo_space = HPOParameterSpace(hyp_config)  # Define HPO space based on hyp_config
    callback_manager = CallbackManager()  # Assuming any needed callbacks are set here

    # Initialize and run HPO Trainer
    hpo_trainer = HPOTrainer(
        model=model,
        data_config=data_config,
        hyp_config=hyp_config,
        hpo_space=hpo_space,
        logger=logger,
        callback_manager=callback_manager
    )
    hpo_trainer.train()

    # Close the logger after training is complete
    logger.close()

if __name__ == "__main__":
    main()
