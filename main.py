from src.inference.inference_pipeline import run_inference
from src.analyze.modules.traj_explorer import PlotXY
from src.analyze.modules.average_visits import AverageVisits
from src.analyze.modules.duration import Duration
import yaml
import argparse

def load_config(config_path):
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)

def main(task_name):
    config = load_config(f"configs/{task_name}.yaml")
    if task_name == 'infer':
        run_inference(config)
    elif task_name == 'analyze':
        explorer = PlotXY(config)
        explorer()
        average_visits = AverageVisits(config)
        average_visits()
        duration = Duration(config)
        duration()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run YOLO inference with tracking or detection.')
    parser.add_argument('--task_name', choices=['infer', 'analyze'], required=False, help='The task to be performed')
    args = parser.parse_args()
    main(args.task_name)

