import os
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_col, theme_classic, labs, geom_point, geom_line ,geom_jitter, geom_errorbar
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
            per_frame.groupby(["time_interval", "teratment_or_rep","image_name"], as_index=False)["trajectory_count"]
            .nunique()
            .reset_index()
        )

        #convert frame to minutes  
        per_frame["time"] = per_frame["image_idx"] / self.fps / 60  # convert to minutes
        
        if self.teratment_or_rep == "treatment":
            per_time_interval = (
                per_frame.groupby(['teratment_or_rep', 'time_interval'])
                .agg(
                    mean_trajectory_count=('trajectory_count', 'mean'),
                    se_trajectory_count=('trajectory_count', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))  # Standard Error
                )
                .reset_index()
                )
        else:
            per_time_interval = (
                per_frame.groupby(['teratment_or_rep', 'time_interval'])
                .agg(
                    mean_trajectory_count=('trajectory_count', 'mean')
                )
                .reset_index()
            )
        count=('trajectory_count', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
                    .reset_index()
            )   
        return averave_visits, per_frame ,per_time_interval




    def save_new_df(self, data,name):
        
        output_csv = os.path.join(self.results_output, f"{name}.csv")
        create_output_dir(self.results_output) 
        data.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    def box_plot_results(self, averave_visits):
        
        mean_data = (
        averave_visits.groupby("teratment_or_rep", as_index=False)
        .agg(
            trajectory_count=("trajectory_count", "mean"),
            sem=("trajectory_count", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))  # Standard Error of Mean
        )
        )

        # Create the plot
        plot = (
            ggplot()
            # Add the bars for the mean values
            + geom_col(mean_data, aes(x="teratment_or_rep", y="trajectory_count", fill="teratment_or_rep"),color = "black")
            # Overlay the individual data points
            + geom_jitter(averave_visits, aes(x="teratment_or_rep", y="trajectory_count"),color="black", width=0.2, alpha=0.8)
            
            + theme_classic()
            + labs(
                title=" ",
                x=" ",
                y="Mean Trajectory Count",
                fill = " "
            )
        )
        if self.teratment_or_rep == "treatment":
            plot += geom_errorbar(
                mean_data,
                aes(x="teratment_or_rep", ymin="trajectory_count", ymax="trajectory_count + sem"),
                color="black",
                width=0.2  # Width of error bar caps
    )

        output_path = os.path.join(self.plot_path, "average_visits.png")
        create_output_dir(self.plot_path)
        plot.save(output_path)
        print(f"Plot saved to {output_path}")

    def time_plot_results(self, per_time_interval):
        
        plot = (
            ggplot(per_time_interval, aes(x="time_interval", y="mean_trajectory_count",color='teratment_or_rep'))
            + geom_point()
            + geom_line()
            + theme_classic()
            + labs(
                title=" ",
                x="Time (minutes)",
                y="Trajectory's number",
                color=" "
            )
        )       
        if self.teratment_or_rep == "treatment":
            plot += geom_errorbar(aes(ymin=per_time_interval["mean_trajectory_count"]-per_time_interval["se_trajectory_count"], 
                                ymax=per_time_interval["mean_trajectory_count"]+per_time_interval["se_trajectory_count"])
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
        average_visits,per_frame, per_time_interval = self._aggregate_trajectories(df)
        self.save_new_df(data = average_visits,name = "average_visits")
        self.save_new_df(data = per_time_interval,name = "per_time_interval")
        self.box_plot_results(average_visits)
        self.time_plot_results(per_time_interval)


# Usage Example:
# average_visits = AverageVisits('config.json')
# average_visits()
