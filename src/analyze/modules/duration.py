import os
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_boxplot, theme_classic, labs, geom_point
from src.utils.config import load_config
from src.utils.common import create_output_dir




class Duration:
    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)
        self.data_path = self.config['analyze']["csv_path"]
        self.plot_path = self.config["analyze"]["plots_dir"]
        self.results_output = self.config["analyze"]["csv_results_dir"]
        self.time_intervals = float(self.config["average_visits"]["time_intervals"])
        self.fps = float(self.config["average_visits"]["fps"])

    def load_data(self):
        return pd.read_csv(self.data_path)

    def calculate_Duration(self, df):
        
        # Calculate duration (in terms of frame indices) for each track_id
        track_durations = (
            df.groupby('track_id')
            .agg(
                min_idx=('image_idx', 'min'),
                max_idx=('image_idx', 'max'),
                image_name=('image_name', 'min')
            )
            .assign(duration=lambda df: df['max_idx'] - df['min_idx'])
            .reset_index()
            .rename(columns={'min_idx': 'min', 'max_idx': 'max', 'duration':'duration(frames)'})
)
        
        return track_durations

    def save_new_df(self, track_durations):
        """
        Save the processed DataFrame to a CSV file.
        """
        create_output_dir(self.results_output)  # Ensure the directory exists
        output_csv = os.path.join(self.results_output, "duration.csv")
        track_durations.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    def plot_results(self, track_durations):
        """
        Plot the trajectory counts as a barplot.
        """
        plot = (
            ggplot(track_durations, aes(x="image_name", y="duration(frames)"))
            + geom_boxplot()
            + geom_point(alpha=0.6)
            + theme_classic()
            + labs(
                title=f"duration for each trajectory",
                x="Treatment",
                y="Duration (Frames)",
            )
        )
        output_path = os.path.join(self.plot_path, "duration.png")
        create_output_dir(self.plot_path)
        plot.save(output_path)
        print(f"Plot saved to {output_path}")

    def __call__(self):
        
        df = self.load_data()
        track_durations = self.calculate_Duration(df)
        self.save_new_df(track_durations)
        self.plot_results(track_durations)


