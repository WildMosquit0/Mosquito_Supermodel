# src/utils/logger.py

import logging
import os
import sys

# Create a global logger object (not yet configured)
logger = logging.getLogger('GlobalLogger')

def setup_logger(config):
    """
    Set up the logger based on the configuration.
    This function should be called once, typically in the main entry point of the application.

    Args:
        config (dict): The configuration dictionary from which logging settings are taken.
    """
    # Avoid configuring the logger more than once
    if logger.hasHandlers():
        return logger

    # Get logging parameters from config
    log_level = config['logging'].get('log_level', 'INFO')
    log_file = config['logging'].get('log_file', None)
    output_dir = config['output'].get('output_dir', './')  # Default to './' if not set

    # If no log file is provided, use the default in the output directory
    if not log_file:
        log_file = os.path.join(output_dir, 'logfile.log')

    # Set the log level
    logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Create file handler if log_file is set
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Define log format
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set format for both handlers
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
