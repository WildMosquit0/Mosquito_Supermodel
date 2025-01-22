from ultralytics import YOLO
import yaml
import os
import pandas as pd
import yaml

def update_yaml_file(file_path, updates):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    data.update(updates)

    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

def validate_all_folders(config_path):

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    model_path = config['case_study']['model_path']
    base_dir = config['case_study']['base_dir']
    yaml_template_path = config['case_study']['yaml_template_path']
    save_dir = config['case_study']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    model = YOLO(model_path)

    metrics_data = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        yaml_path = os.path.join(save_dir, f"val_{folder}.yaml")
        updates = {
            'path': folder_path,
            'val': 'images/val',
            'train' : 'null',
            'names': {0: 'm'},
            'nc': 1
        }
        update_yaml_file(yaml_template_path, updates)

        with open(yaml_path, 'w') as yaml_file:
            yaml.safe_dump(updates, yaml_file)

        metrics = model.val(data=yaml_path)

        metrics_data.append({
            'Folder': folder,
            'mAP50-95': metrics.box.map,
            'mAP50': metrics.box.map50,
            'mAP75': metrics.box.map75,
             'Precision': metrics.box.p,  
            'Recall': metrics.box.r


        })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(save_dir, "validation_metrics.csv"), index=False)
    print(f"Metrics saved to {os.path.join(save_dir, 'validation_metrics.csv')}")


config_path = "/home/wildmosquit0/git/Mosquito_Supermodel/src/figures/config.yaml"


validate_all_folders(config_path)
