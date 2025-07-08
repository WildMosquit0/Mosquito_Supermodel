import yaml
import os
import sys
import argparse
import cProfile
import pstats
import builtins
import logging

from src.postprocess.infer_sngle_or_multi import inference_single_or_multi
from src.analyze.analysis_pipeline import run_analysis
from src.utils.config_ops import load_config
from src.logger.main_logger import setup_logger, setup_file_logging

opened_files = set()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

logger = setup_file_logging(PROJECT_ROOT, opened_files)

def main(task: str) -> None:
    conf_yaml_path = os.path.abspath(f"configs/{task}.yaml")
    config = load_config(conf_yaml_path)

    logger.info(f"Loaded config: {conf_yaml_path}")
    logger.info(f"Starting task: {task}")

    if task == 'infer':
        inference_single_or_multi(config, conf_yaml_path, logger)
    elif task == 'analyze':
        run_analysis(config, conf_yaml_path, logger)

    logger.info(f"Task {task} completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run YOLO inference with tracking, detection, or slicing.')
    parser.add_argument('--task', choices=['infer', 'analyze'], default='infer', help='The task to be performed')
    args = parser.parse_args()

    conf_yaml_path = os.path.abspath(f"configs/{args.task}.yaml")
    config = load_config(conf_yaml_path)
    setup_logger(config['output_dir'], logger)
    main(args.task)
