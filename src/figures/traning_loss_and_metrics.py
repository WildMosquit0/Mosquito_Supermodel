import pandas as pd
import yaml
from plotnine import ggplot, aes, geom_line, labs, theme_minimal
import os

def plot_yolo_training_results_from_yaml(config_file):
    # Load configuration from YAML file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    results_file = config['loss_figure']['csv_path']
    save_dir = config['loss_figure']['save_folder']

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Read the results.txt file into a DataFrame
    df = pd.read_csv(results_file)

    # Plot each loss (train and validation) separately
    loss_types = ["box_loss", "cls_loss", "dfl_loss"]
    for loss in loss_types:
        loss_plot = (
            ggplot(df)
            + aes(x='epoch')
            + geom_line(aes(y=f'train/{loss}'), color='blue')
            + geom_line(aes(y=f'val/{loss}'), color='red')
            + labs(title=f'{loss.capitalize()} Over Epochs', x='Epoch', y='Loss')
        )
        loss_plot.save(filename=f"{save_dir}/{loss}_over_epochs.png", dpi=300)

    # Plot each metric separately
    metrics = ["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"]
    for metric in metrics:
        metric_plot = (
            ggplot(df)
            + aes(x='epoch')
            + geom_line(aes(y=f'metrics/{metric}'), color='black')
            + labs(title=f' ', x='Epoch', y=f'{metric}')
            
        )
        metric_plot.save(filename=f"{save_dir}/{metric}_over_epochs.png", dpi=300)

    print(f"Plots saved to {save_dir}")

# Example usage
config_file = "/home/wildmosquit0/git/Mosquito_Supermodel/src/figures/config.yaml"
plot_yolo_training_results_from_yaml(config_file)
