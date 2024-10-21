import json
import os

def load_config(config_file="config.json"):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def copy_config(config):
    output_dir = config.get('output').get('output_dir')
    output_path = os.path.join(output_dir, 'config.json')
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

