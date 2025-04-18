import yaml
import argparse
import cProfile
import pstats
from src.inference.inference_pipeline import run_inference
from src.analyze.analysis_pipeline import run_analysis  # Ensure you have this implemented
from src.utils.config import load_config
import os

def main(task_name: str) -> None:
    config = load_config(f"configs/{task_name}.yaml")
    if task_name == 'infer':
        source = config.get('images_dir', '')
        flag = os.path.basename(source)
        if flag.find(".") <= 0:
            vid_names = os.listdir(config.get("images_dir",""))
            for vid_name in vid_names:
                    export_name = vid_name.split(".")[0]
                    config["images_dir"] = os.path.join(source, vid_name)
                    config["csv_filename"] = f"{export_name}_results.csv"
                    run_inference(config)
        else:
            run_inference(config)

    elif task_name == 'analyze':
        run_analysis(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run YOLO inference with tracking, detection, or slicing.')
    parser.add_argument('--task_name', choices=['infer', 'analyze'], default='infer',  help='The task to be performed')
    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        main(args.task_name)
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumtime')
        stats.print_stats(10)
