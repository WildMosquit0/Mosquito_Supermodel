import os
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_col, theme_classic, labs, geom_point, geom_line ,geom_jitter, geom_errorbar,scale_x_continuous
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

        #convert frame to minutes  
        per_frame["time"] = per_frame["image_idx"] / self.fps / 60  # convert to minutes
    
        # calculates each all reps as treatment
        average_visits = (
                per_frame.groupby(['time_interval',"image_name"])
                .agg(
                    mean_trajectory_count=('trajectory_count', 'mean')
                )
                .reset_index()
            )
        average_visits["treatment"] = average_visits["image_name"].str.split("_").str[0]
        
        return average_visits

    def create_summary_stats_for_graphs(self,average_visits):

        summary_bar = (
            average_visits.groupby(self.teratment_or_rep)["mean_trajectory_count"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary_bar["se"] = summary_bar["std"] / summary_bar["count"]**0.5
        summary_bar["mean_upper_se"] = summary_bar["mean"] + summary_bar["se"]
        
        summary_time = (
            average_visits.groupby([self.teratment_or_rep,"time_interval"])["mean_trajectory_count"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary_time["se"] = summary_time["std"] / summary_time["count"]**0.5
        summary_time["mean_upper_se"] = summary_time["mean"] + summary_time["se"]
        summary_time["mean_lower_se"] = summary_time["mean"] - summary_time["se"]

        return summary_bar, summary_time




    def save_new_df(self, data,name):
        
        output_csv = os.path.join(self.results_output, f"{name}.csv")
        create_output_dir(self.results_output) 
        data.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    def bar_plot_results(self, average_visits,summary_bar):
        
        # Create the plot
        plot = (
            ggplot()
            + geom_col(
                summary_bar,
                aes(x=self.teratment_or_rep, y="mean", fill=self.teratment_or_rep),
                color="black"
            )
            + geom_jitter(
                average_visits,
                aes(x=self.teratment_or_rep, y="mean_trajectory_count"),
                color="black",
                width=0.2,
                alpha=0.8
            )
            + theme_classic()
            + labs(
                title=" ",
                x=" ",
                y="Mean Trajectory Count",
                fill=" "
            )
        )

        # Add error bars if applicable
        if self.teratment_or_rep == "treatment":
            plot += geom_errorbar(
                summary_bar,
                aes(
                    x=self.teratment_or_rep,
                    ymin="mean",
                    ymax="mean_upper_se"
                ),
                width=0.2
            )
       

        output_path = os.path.join(self.plot_path, "average_visits.png")
        create_output_dir(self.plot_path)
        plot.save(output_path)
        print(f"Plot saved to {output_path}")

    def time_plot_results(self, average_visits,summary_time):
        
        plot = (
            ggplot(summary_time, aes(x="time_interval", y="mean",color=self.teratment_or_rep))
            + geom_point()
            + geom_line()
            + theme_classic()
            + labs(
                title=" ",
                x="Time (minutes)",
                y="Trajectory's number",
                color=" "
            )
            + scale_x_continuous(
                limits=[0, average_visits["time_interval"].max()],
                breaks=range(0, average_visits["time_interval"].max() + 1,3)
            )
        )       
        if self.teratment_or_rep == "treatment":
            plot += geom_errorbar(
                summary_time,
                aes(
                    x="time_interval",
                    ymin="mean_lower_se",
                    ymax="mean_upper_se"
                ),
                width=0.2
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
        average_visits = self._aggregate_trajectories(df)
        summary_bar, summary_time = self.create_summary_stats_for_graphs(average_visits)
        self.save_new_df(data = average_visits,name = "average_visits")
        #self.save_new_df(data = per_time_interval,name = "per_time_interval")
        self.bar_plot_results(average_visits,summary_bar)
        self.time_plot_results(average_visits, summary_time)


# Usage Example:
# average_visits = AverageVisits('config.json')
# average_visits()
