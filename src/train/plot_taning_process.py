from plotnine import ggplot, aes, geom_line, labs, theme_classic, scale_color_manual
import pandas as pd
import os
# Example: Plot training and validation box loss over epochs

class MetricPlotter:
    def __init__(self, data):
        """
        Initialize the MetricPlotter with a DataFrame.
        :param data: DataFrame containing the metrics and epochs.
        """
        self.data = pd.read_csv(path)
        self.clean_data()

    def clean_data(self):
        """
        Clean and prepare the data (strip column names).
        """
        # Strip leading/trailing spaces from column names
        self.data.columns = self.data.columns.str.strip()

    def prepare_data(self, metrics):
        """
        Prepare data for the given list of metrics by melting the DataFrame.
        :param metrics: List of metric column names to include.
        :return: Melted DataFrame ready for plotting.
        """
        return self.data.melt(
            id_vars=['epoch'],
            value_vars=metrics,
            var_name='metric',
            value_name='value'
        )

    def create_combined_loss_plots(self, output_dir="loss_plots"):
        """
        Create and save combined plots for train and validation loss metrics.
        :param output_dir: Directory where plots will be saved.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define loss pairs (train/val)
        loss_pairs = [
            ('train/box_loss', 'val/box_loss'),
            ('train/cls_loss', 'val/cls_loss'),
            ('train/dfl_loss', 'val/dfl_loss')
        ]

        for train_metric, val_metric in loss_pairs:
            # Prepare the data for these metrics
            melted_data = self.prepare_data([train_metric, val_metric])

            # Create the plot
            plot = (
                ggplot(melted_data, aes(x='epoch', y='value', color='metric'))
                + geom_line(size=1)
                + theme_classic()
                + labs(
                    title=f"{train_metric.split('/')[1]} Loss (Train vs. Validation)",
                    x="Epoch",
                    y=f"{train_metric.split('/')[1]} Loss",
                    color="Metric"
                )
                + scale_color_manual(
                    values={
                        train_metric: 'blue',
                        val_metric: 'red'
                    }
                )
            )

            # Save the plot
            filename = f"{output_dir}/{train_metric.split('/')[1]}_loss_plot.png"
            plot.save(filename)
            print(f"Plot saved as {filename}")

    def create_separate_metric_plots(self, output_dir="metric_plots"):
        """
        Create and save separate plots for precision, recall, and mAP metrics.
        :param output_dir: Directory where plots will be saved.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define individual metrics
        metrics = [
            'metrics/precision(B)',
            'metrics/recall(B)',
            'metrics/mAP50(B)',
            'metrics/mAP50-95(B)'
        ]

        for metric in metrics:
            # Prepare the data for this metric
            melted_data = self.prepare_data([metric])

            # Create the plot
            plot = (
                ggplot(melted_data, aes(x='epoch', y='value'))
                + geom_line(color='blue', size=1)
                + theme_classic()
                + labs(
                    title=f"{metric.split('/')[1]} Over Epochs",
                    x="Epoch",
                    y=f"{metric.split('/')[1]}"
                )
            )

            # Save the plot
            filename = f"{output_dir}/{metric.replace('/', '_')}_plot.png"
            plot.save(filename)
            print(f"Plot saved as {filename}")

path = "/home/bohbot/Evyatar/runs/detect/most_update_batch3/results.csv"

plotter = MetricPlotter(path)


# Create and save combined plots for losses (Train + Validation together)
plotter.create_combined_loss_plots()

# Create and save separate plots for metrics
plotter.create_separate_metric_plots()
