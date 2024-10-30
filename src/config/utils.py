import json
import os
import yaml
from typing import Dict


def load_config(config_file="config.json"):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def copy_config(config):
    output_dir = config.get('output').get('output_dir')
    output_path = os.path.join(output_dir, 'config.json')
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)


class ConfigLoader:
    def __init__(self, data_config_path: str, hyp_config_path: str = None):
        self.data_config_path = data_config_path
        self.hyp_config_path = hyp_config_path

    def load_yaml(self, file_path: str) -> Dict:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def get_data_config(self) -> Dict:
        return self.load_yaml(self.data_config_path)

    def get_hyp_config(self) -> Dict:
        return self.load_yaml(self.hyp_config_path) if self.hyp_config_path else {}
