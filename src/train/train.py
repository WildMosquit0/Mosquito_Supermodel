from ultralytics import YOLO
import yaml


class YOLOTrainer:
    """
    YOLO Trainer for standard and hyperparameter optimization (HPO) training.
    """

    def __init__(self, config_path: str, hyp_path: str = None, use_hpo: bool = False):
        self.model = YOLO('yolo11n.pt')  # Initialize YOLO with weights
        self.data_config_path = config_path  # Path to the data YAML file
        self.hyp_config_path = hyp_path  # Path to the hyp YAML file, if any
        self.use_hpo = use_hpo

    def train(self):
        if self.use_hpo:
            # For HPO, use Ultralytics' `tune` method
            self.model.tune(
                data=self.data_config_path,  # Path to the data YAML file
                epochs=self.load_yaml(self.hyp_config_path)['hpo_params']['epochs'],
                iterations=self.load_yaml(self.hyp_config_path)['hpo_params']['iterations'],
                optimizer=self.load_yaml(self.hyp_config_path)['hpo_params']['optimizer'],
                lr0=self.load_yaml(self.hyp_config_path)['lr0'],
                batch=self.load_yaml(self.hyp_config_path)['batch_size'],
                save=True,
                plots=False,
                val=False
            )
        else:
            # For standard training
            self.model.train(
                data=self.data_config_path,  # Path to the data YAML file
                epochs=self.load_yaml(self.data_config_path)['training']['epochs'],
                batch=self.load_yaml(self.data_config_path)['training']['batch_size'],
                imgsz=self.load_yaml(self.data_config_path)['training']['imgsz'],
                optimizer=self.load_yaml(self.data_config_path)['training']['optimizer'],
                save_dir=self.load_yaml(self.data_config_path)['output']['output_dir'],
            )

    def load_yaml(self, file_path: str):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)


if __name__ == '__main__':
    trainer = YOLOTrainer('data.yaml', 'hyp.yaml', use_hpo=False)
    trainer.train()

    # trainer = YOLOTrainer('data.yaml', 'hpo.yaml', use_hpo=True)
    # trainer.train()