import yaml
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)

def create_data_yaml(data_dict: Dict[str, Any]) -> str:
    """Creates a temporary YAML file from the provided data dictionary."""
    data_yaml_path = 'temp_data.yaml'
    try:
        with open(data_yaml_path, 'w') as file:
            yaml.dump(data_dict, file)
        logging.info(f"Data YAML created at {data_yaml_path}")
    except Exception as e:
        logging.error(f"Failed to create data YAML: {e}")
        raise
    return data_yaml_path
