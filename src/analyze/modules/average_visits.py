import os
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_boxplot, theme_classic, labs, geom_point
from src.utils.config import load_config
from src.utils.common import create_output_dir


class AverageVisits:
    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)
        self.data_path = self.config['analyze']["csv_path"]
        self.plot_path = self.config["analyze"]["plots_dir"]
        self.results_output = self.config["analyze"]["csv_results_dir"]
        self.time_intervals = float(self.config["average_visits"]["time_intervals"])
        self.fps = float(self.config["average_visits"]["fps"])

    def _load_data(self):
        return pd.read_csv(self.data_path)

    def _calculate_time_intervals(self, df):
        """
        Calculate the time intervals for each frame based on fps and time_intervals in seconds.
        """
        # Convert `image_idx` to a NumPy array
        df["image_idx"] = pd.to_numeric(df["image_idx"], errors="coerce")
        if df["image_idx"].isnull().any():
            print("Warning: Some `image_idx` values were non-numeric and have been removed.")
        df = df.dropna(subset=["image_idx"])

        # Calculate time intervals
        divisor = self.fps * self.time_intervals
        if divisor == 0:
            raise ValueError("Divisor (fps * time_intervals) cannot be zero.")

        df["time_interval"] = (df["image_idx"] / divisor).astype(int)
        print("Calculated time intervals based on seconds.")
        return df

    def _aggregate_trajectories(self, df):
        return (
            df.groupby(["time_interval", "image_name"])["track_id"]
            .nunique()  # Count unique `track_id` values per interval
            .reset_index(name="trajectory_count")
        )

    def save_new_df(self, df):
        
        output_csv = os.path.join(self.results_output, "average_visits.csv")
        create_output_dir(self.results_output) 
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    def _plot_results(self, df):
        
        plot = (
            ggplot(df, aes(x="image_name", y="trajectory_count"))
            + geom_boxplot()
            + geom_point(alpha=0.6)
            + theme_classic()
            + labs(
                title=" ",
                x="Image Name",
                y="Trajectory Count",
            )
        )
        output_path = os.path.join(self.plot_path, "average_visits.png")
        create_output_dir(self.plot_path)
        plot.save(output_path)
        print(f"Plot saved to {output_path}")

    def __call__(self):
        """
        Execute the pipeline: load data, calculate intervals, save results, and plot.
        """
        df = self._load_data()
        df = self._calculate_time_intervals(df)
        aggregated_df = self._aggregate_trajectories(df)
        self.save_new_df(aggregated_df)
        self._plot_results(aggregated_df)


# Usage Example:
# average_visits = AverageVisits('config.json')
# average_visits()
