from ultralytics import YOLO
import yaml

class YOLOHPO:
    """
    A class to handle YOLO model hyperparameter tuning.
    """

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = YOLO(self.config['model']['weights'])

    def tune(self):
        self.model.tune(
            data=self.config['data'],
            epochs=self.config['hpo_params']['epochs'],
            iterations=self.config['hpo_params']['iterations'],
            optimizer=self.config['hpo_params']['optimizer'],
            plots=False,
            save=False,
            val=False
        )

if __name__ == "__main__":
    tuner = YOLOHPO(config_path="hpo.yaml")
    tuner.tune()
