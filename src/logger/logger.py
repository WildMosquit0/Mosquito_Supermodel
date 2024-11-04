import logging
import os
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any


logger = logging.getLogger('GlobalLogger')

def setup_logger(output_dir: str) -> logging.Logger:
    logger = logging.getLogger('GlobalLogger')
    logger.setLevel(logging.INFO)

    log_file = os.path.join(output_dir, 'train.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger

class Logger:
    """Handles logging of hyperparameters and metrics for experiments."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, trial_number: int, metrics: Any) -> None:
        """Logs metrics from a training run."""
        metrics_dict = {'fitness': metrics.fitness, 'precision': metrics.box.map50, 'recall': metrics.box.map50_95}
        for key, value in metrics_dict.items():
            self.writer.add_scalar(f"Trial_{trial_number}/{key}", value)

    def close(self) -> None:
        """Closes the TensorBoard writer."""
        self.writer.close()