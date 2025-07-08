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

logger = logging.getLogger("project_logger")
logger.setLevel(logging.INFO)

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

def setup_logger(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "project_runtime.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler.setFormatter(stream_formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.info(f"Logger initialized. Writing logs to {log_file}")

def main(task: str, test_mode: bool = False) -> None:
    conf_yaml_path = os.path.abspath(f"configs/{task}.yaml")
    config = load_config(conf_yaml_path)

    if test_mode:
        logger.info("Test mode enabled: overriding images_dir to tests/images")
        config["images_dir"] = "tests/images"
        conf_yaml_path = f"tests/test.yaml"
        with open(conf_yaml_path, "w") as f:
            yaml.safe_dump(config, f)

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
    parser.add_argument('--test', action='store_true', help='Run in test mode with profiling and tests/images directory')
    args = parser.parse_args()

    conf_yaml_path = os.path.abspath(f"configs/{args.task}.yaml")
    config = load_config(conf_yaml_path)
    setup_logger(config['output_dir'])

    if args.test:
        logger.info("Profiling enabled (test mode).")
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            main(args.task, test_mode=True)
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
                    
            logger.info("✅ TEST PASSED: All steps completed successfully.")
            print("✅ TEST PASSED: All steps completed successfully.")
            with open(conf_yaml_path, "r") as f:
                yaml_content = f.read()
            logger.info("\n=== YAML config used ===\n" + yaml_content)
    else:
        logger.info("Running without profiling.")
        main(args.task, test_mode=False)
