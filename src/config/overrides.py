class YOLOOverrides:
    """
    Handles configuration overrides for the YOLO model.
    """
    
    def __init__(self, config: dict):
        self.config = config

    def get_overrides(self) -> dict:
        # Ensure 'data' is a string pointing to dataset locations (train/val paths or YAML file)
        return {
            'data': f"{self.config['input']['train']},{self.config['input']['val']}",
            'epochs': self.config['training']['epochs'],
            'imgsz': self.config['training'].get('img_size', 640),
            'lr0': self.config['training']['lr'],
            'optimizer': self.config['training'].get('optimizer', 'adam'),
            'save_dir': self.config['output']['output_dir']
        }
