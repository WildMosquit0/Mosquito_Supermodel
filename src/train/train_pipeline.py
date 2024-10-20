from src.utils.config import load_config, copy_config
from src.utils.logger import setup_logger, logger
from src.train.trainer import TrainerWrapper
from src.train.hpo import HPO

def run_training():
    # Load the configuration from the config file
    config = load_config("config.json")
    copy_config(config)

    # Set up the logger with the config
    setup_logger(config)
    logger.info("Starting standard training...")

    # Initialize the trainer
    trainer_wrapper = TrainerWrapper(config)

    # Train the model
    trainer_wrapper.train()

def run_hpo():
    logger.info("Starting hyperparameter optimization (HPO)...")

    # Load the configuration from the config file
    config = load_config("config.json")

    # Set up the logger with the config
    logger = setup_logger(config)
    logger.info("Starting hyperparameter optimization (HPO)...")

    # Initialize HPO process
    hpo = HPO(config)

    # Run hyperparameter optimization
    hpo.optimize_hyperparameters()

if __name__ == "__main__":
    # You can switch between regular training and HPO here
    run_training()  # For standard training
    # run_hpo()  # For hyperparameter optimization
