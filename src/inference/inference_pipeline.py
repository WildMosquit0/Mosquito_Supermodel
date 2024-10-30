import argparse
from typing import Any, Dict
from src.config.utils import load_config
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    inferer = Inferer(config=config)
    results = inferer.infer()

    parser = ResultsParser(results=results, config=config)

    parser.parse_and_save()


if __name__ == "__main__":
    main()
