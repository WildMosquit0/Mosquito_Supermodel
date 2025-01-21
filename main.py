import yaml
import argparse

from src.inference.inference_pipeline import run_inference
from src.analyze.analysis_pipeline import run_analysis
from src.utils.config import load_config

def main(task_name):
    config = load_config(f"configs/{task_name}.yaml")
    if task_name == 'infer':
        run_inference(config)
    elif task_name == 'analyze':
        run_analysis(config)


import cProfile
import pstats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run YOLO inference with tracking or detection.')
    parser.add_argument('--task_name', choices=['infer', 'analyze'], required=False, help='The task to be performed')
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

