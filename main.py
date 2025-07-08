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

opened_files = set()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

_original_open = builtins.open
def logged_open(file, *args, **kwargs):
    abs_path = os.path.abspath(file)
    if abs_path.startswith(PROJECT_ROOT):
        rel_path = os.path.relpath(abs_path, PROJECT_ROOT)
        if rel_path not in opened_files:
            logger.info(f"Accessed file: {rel_path}")
            opened_files.add(rel_path)
    return _original_open(file, *args, **kwargs)

builtins.open = logged_open

def main(task: str) -> None:
    conf_yaml_path = os.path.abspath(f"configs/{task}.yaml")
    config = load_config(conf_yaml_path)
    os.makedirs(config['output_dir'], exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config['output_dir'], "project_runtime.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
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

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        main(args.task)
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumtime')
        logger.info("\n=== Top 10 slowest functions ===")
        stats.print_stats(10)
        logger.info("\n=== Python files from my project ===")
        for module in sorted(sys.modules.keys()):
            mod = sys.modules[module]
            filepath = getattr(mod, '__file__', None)
            if filepath and filepath.endswith(".py") and filepath.startswith(PROJECT_ROOT):
                rel_path = os.path.relpath(filepath, PROJECT_ROOT)
                logger.info(f"Loaded: {rel_path}")
        if opened_files:
            logger.info("\n=== Other files from my project accessed ===")
            for f in sorted(opened_files):
                logger.info(f"Accessed: {f}")
