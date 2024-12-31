import os
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_boxplot, theme_classic, labs, geom_point, geom_smooth,geom_jitter
from src.utils.config import load_config
from src.utils.common import create_output_dir


class AverageVisits:
    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)
        self.plot_path = self.config["analyze"]["plots_dir"]
        self.results_output = self.config["analyze"]["csv_results_dir"]
        self.time_intervals = float(self.config["average_visits"]["time_intervals"])
        self.fps = float(self.config["average_visits"]["fps"])
        

        self.data_path = self.config['analyze']["csv_path"]
        self.teratment_or_rep = self.config['analyze']['teratment_or_rep']
        
    def _load_data(self):
        return pd.read_csv(self.data_path)
    
    def _does_it_teratment_or_rep(self,df):
        if self.teratment_or_rep not in df.columns:
            
            raise KeyError("Error: The column " ,self.teratment_or_rep, " does not exist in the DataFrame.")
        
        df['teratment_or_rep'] = df[self.teratment_or_rep].astype('str')
        return df
        

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
        
        # Step 1: Calculate unique track IDs per frame
        per_frame = (
            df.groupby(["image_idx", "image_name", "teratment_or_rep", "time_interval"])["track_id"]
            .unique()
            .reset_index()
        )
        per_frame["trajectory_count"] = per_frame["track_id"].apply(len)
        
        # Step 2: Aggregate these counts by time intervals and treatment/replicate
        averave_visits = (
            per_frame.groupby(["time_interval", "teratment_or_rep"], as_index=False)["trajectory_count"]
            .mean()
        )

        #convert frame to minutes  
        per_frame["time"] = per_frame["image_idx"] / self.fps / 60  # convert to minutes
        
        per_time_interval = (
            per_frame.groupby(['teratment_or_rep', 'time_interval'])
            .agg(mean_trajectory_count=('trajectory_count', 'mean'),
                se_trajectory_count=('trajectory_count', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))))
            .reset_index()
)
        return averave_visits, per_frame ,per_time_interval




    def save_new_df(self, data,name):
        
        output_csv = os.path.join(self.results_output, f"{name}.csv")
        create_output_dir(self.results_output) 
        data.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    def box_plot_results(self, averave_visits):
        
        plot = (
            ggplot(averave_visits, aes(x="teratment_or_rep", y="trajectory_count"))
            + geom_boxplot()
            + geom_jitter(alpha=0.6)
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

    def time_plot_results(self, per_frame):
        
        plot = (
            ggplot(per_frame, aes(x="time", y="trajectory_count",color='teratment_or_rep'))
            + geom_smooth(method='loess', span=0.1,se=True)
            + theme_classic()
            + labs(
                title=" ",
                x="time (minutes)",
                y="Trajectory's number",
            )
        )
        output_path = os.path.join(self.plot_path, "average_visits_time.png")
        create_output_dir(self.plot_path)
        plot.save(output_path)
        print(f"Plot saved to {output_path}")

    def __call__(self):
        """
        Execute the pipeline: load data, calculate intervals, save results, and plot.
        """
        df = self._load_data()
        df = self._does_it_teratment_or_rep(df)
        df = self._calculate_time_intervals(df)
        average_visits,detections_vs_time, per_time_interval = self._aggregate_trajectories(df)
        self.save_new_df(data = average_visits,name = "average_visits")
        self.save_new_df(data = per_time_interval,name = "visits_vs_frames")
        self.box_plot_results(average_visits)
        self.time_plot_results(per_time_interval)


# Usage Example:
# average_visits = AverageVisits('config.json')
# average_visits()
