import argparse
from typing import Any, Dict
from src.utils.config import load_config
from src.inference.inferer import Inferer
from src.postprocess.saver import ResultsParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO inference with tracking or detection."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.json",
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()

def main(config_path=None):
    """
    Main function to perform inference and save results.
    If config_path is provided, it uses that instead of parsing command-line arguments.
    """
    if config_path is None:
        args = parse_args()  # Use command-line arguments if config_path is not provided
        config_path = args.config

    # Load configuration
    config = load_config(config_path)
    inferer = Inferer(config=config)
    results = inferer.infer()
    # Parse and save results
    parser = ResultsParser(results=results, config=config)
    parser.parse_and_save()
