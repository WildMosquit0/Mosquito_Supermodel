# src/utils/logger.py

import logging
import os
import sys

# Create a global logger object (not yet configured)
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

    return logger
