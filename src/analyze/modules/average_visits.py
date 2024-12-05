import os
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_bar, geom_point, theme_minimal, labs, theme_classic
from src.utils.config import load_config
from src.utils.common import create_output_dir


class AverageVisits:
    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)
        self.data_path = self.config['analyze']["csv_path"]
        self.plot_path = self.config["analyze"]["output_dir"]
        self.time_intervals = float(self.config["average_visits"]["time_intervals"])
        self.fps = float(self.config["average_visits"]["fps"])

    def _load_data(self):
        return pd.read_csv(self.data_path)

    def _calculate_time_intervals(self, df):
        """
        Calculate the time intervals for the dataframe.
        Args:
            df (pd.DataFrame): Dataframe with a valid `image_idx` column.
        Returns:
            pd.DataFrame: Dataframe with an additional `time_interval` column.
        """
        # Convert `image_idx` to a NumPy array
        image_idx = pd.to_numeric(df["image_idx"], errors="coerce").to_numpy()
        if np.isnan(image_idx).any():
            print("Warning: Some `image_idx` values were non-numeric and have been removed.")
        image_idx = image_idx[~np.isnan(image_idx)]

        # Determine if `time_intervals` is in minutes
        is_minutes = self.config["average_visits"].get("interval_unit", "seconds") == "minutes"
        divisor = self.fps * self.time_intervals * (60 if is_minutes else 1)

        if divisor == 0:
            raise ValueError("Divisor (fps * time_intervals) cannot be zero.")

        # Perform the calculation
        time_intervals = (image_idx / divisor).astype(int)
        df = df.iloc[:len(image_idx)]
        df["time_interval"] = time_intervals
        print(f"Calculated time intervals based on {'minutes' if is_minutes else 'seconds'}.")
        return df


    def _aggregate_trajectories(self, df):
        """
        Aggregate the trajectory counts per time interval.
        """
        return (
            df.groupby(["time_interval", "image_name"])["track_id"]
            .nunique()  # Count unique `track_id` values per interval
            .reset_index(name="trajectory_count")
        )

    def _plot_results(self, df):
        plot = (
            ggplot(df, aes(x="image_name", y="trajectory_count"))
            + geom_bar(stat="identity", fill="skyblue", alpha=0.7)
            + theme_classic()
            + labs(
                title=f"Average Visits Per {self.time_intervals} Minutes",
                x="Time Interval",
                y="Trajectory Count",
            )
        )
        create_output_dir(self.plot_path)
        output_path = os.path.join(self.plot_path, "average_visits.png")
        plot.save(output_path)
        print(f"Plot saved to {output_path}")

    def __call__(self):
        df = self._load_data()
        df = self._calculate_time_intervals(df)
        aggregated_df = self._aggregate_trajectories(df)
        self._plot_results(aggregated_df)


# Usage Example:
# average_visits = AverageVisits('config.json')
# average_visits()
