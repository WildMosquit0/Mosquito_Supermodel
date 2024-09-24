import argparse
from typing import Any, Dict
from src.utils.config import load_config
from src.inference.inferer import Inferer
from src.postprocess.saver import ResultsParser

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference with tracking or detection.")
    parser.add_argument('--config', type=str, default="config.json", help='Path to the JSON configuration file.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    config: Dict[str, Any] = load_config(args.config)

    inferer = Inferer(
        model_path=config["model"].get("weights"),
        task=config["model"].get("task"),             
        output_dir=config["output"].get("output_dir"),
        images_dir=config["input"].get("images_dir"),         
        save_animations=config["output"].get("save_animations") 
    )
    
    results = inferer.infer()

    parser = ResultsParser(
        results=results, 
        output_dir=inferer.output_dir, 
        csv_filename=config["output"].get("csv_filename"),
        task=config["model"].get("task")
    )
    
    parser.parse_and_save()  

if __name__ == "__main__":
    main()
