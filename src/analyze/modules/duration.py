import os
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_boxplot, theme_classic, labs, geom_jitter, geom_errorbar, geom_point, geom_line, scale_x_continuous
from src.utils.common import create_output_dir

class Duration:
    def __init__(self, config):
        self.config = config
        self.plot_path = f"{self.config['output_dir']}/plots"
        self.fps = float(self.config["settings"]["fps"])
        self.time_intervals = float(self.config.get("settings", {}).get("time_intervals", 1))  
        self.interval_unit = self.config.get("settings", {}).get("interval_unit", "minutes")
        self.filter_time_intervals = self.config.get('settings', {}).get('filter_time_intervals', float('inf'))
        self.data_path = self.config["input_csv"]
        self.treatment_or_rep = self.config['plotxy']['treatment_or_rep']

    def _load_data(self):
        return pd.read_csv(self.data_path)

    def _does_it_treatment_or_rep(self, df):
        if self.treatment_or_rep not in df.columns:
            raise KeyError(f"Error: The column {self.treatment_or_rep} does not exist in the DataFrame.")
        return df

    def _calculate_time_intervals(self, df):
        """Calculate time intervals for each frame"""
        df["image_idx"] = pd.to_numeric(df["image_idx"], errors="coerce")
        df = df.dropna(subset=["image_idx"])
        
        # Convert to the desired time unit
        conversion = 60 if self.interval_unit == "minutes" else 1
        time_interval_length = self.time_intervals * conversion
        
        # Calculate time interval
        df["time_interval"] = ((df["image_idx"] / self.fps) / time_interval_length).astype(int) * self.time_intervals
        
        return df

    def _calculate_duration(self, df):
        """Calculate the duration each mosquito (track_id) is present in the video"""
        # Ensure necessary columns are numeric
        df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")
        df["image_idx"] = pd.to_numeric(df["image_idx"], errors="coerce")
        df = df.dropna(subset=["track_id", "image_idx"])
        
        # Create a clean subset with only necessary columns
        x = df[["track_id", "image_name", "image_idx", self.treatment_or_rep, "time_interval"]]
        
        # Sort by track_id, image_name, and image_idx to ensure proper frame order
        x = x.sort_values(["track_id", "image_name", "image_idx"])
        
        # Calculate duration for each track, ensuring proper separation by image_name
        duration_data = x.groupby(["track_id", "image_name"]).agg(
            start_frame=("image_idx", "min"),
            end_frame=("image_idx", "max"),
            start_time_interval=("time_interval", "min"),
            end_time_interval=("time_interval", "max"),
            treatment=(self.treatment_or_rep, "first")
        ).reset_index()
        
        # Calculate duration in seconds
        duration_data["duration_frames"] = duration_data["end_frame"] - duration_data["start_frame"] + 1
        duration_data["duration_seconds"] = duration_data["duration_frames"] / self.fps
        
        # Rename treatment column back to the original name (treatment_or_rep)
        duration_data = duration_data.rename(columns={"treatment": self.treatment_or_rep})
        
        # For time interval aggregation, we'll use the start time interval
        duration_data["time_interval"] = duration_data["start_time_interval"]
        #duration_data = duration_data[duration_data["duration_seconds"] <= 25]
        return duration_data

    def _aggregate_by_time_intervals(self, duration_data):
        """Aggregate duration data by time intervals"""
        # First, create a detailed dataset with all tracks and their durations per time interval
        all_data = []
        
        for _, row in duration_data.iterrows():
            track_id = row['track_id']
            image_name = row['image_name']
            treatment = row[self.treatment_or_rep]
            duration = row['duration_seconds']
            time_interval = row['time_interval']
            
            all_data.append({
                'track_id': track_id,
                'image_name': image_name,
                self.treatment_or_rep: treatment,
                'time_interval': time_interval,
                'duration_seconds': duration
            })
        
        detailed_df = pd.DataFrame(all_data)
        
        # Now aggregate by time_interval, image_name, and treatment
        time_series_data = (
            detailed_df.groupby(['time_interval', 'image_name', self.treatment_or_rep])
            .agg(
                sum_duration=('duration_seconds', 'sum'),
                track_count=('track_id', 'nunique')
            )
            .reset_index()
        )
        
        # Get valid image_name and treatment_or_rep combinations
        valid_combinations = duration_data[["image_name", self.treatment_or_rep]].drop_duplicates()
        
        # Ensure all time intervals are represented
        max_interval = int(time_series_data["time_interval"].max())
        all_intervals = pd.DataFrame({
            "time_interval": np.arange(0, max_interval + self.time_intervals, self.time_intervals)
        })
        
        # Create a proper grid that only includes valid image-treatment combinations
        combination_grid = []
        for _, row in valid_combinations.iterrows():
            for interval in all_intervals["time_interval"]:
                combination_grid.append({
                    "time_interval": interval,
                    "image_name": row["image_name"],
                    self.treatment_or_rep: row[self.treatment_or_rep]
                })
        
        complete_grid = pd.DataFrame(combination_grid)
        
        # Merge with actual data
        time_series_filled = complete_grid.merge(
            time_series_data, 
            on=['time_interval', 'image_name', self.treatment_or_rep], 
            how='left'
        )
        
        # Fill missing values with 0
        time_series_filled['sum_duration'] = time_series_filled['sum_duration'].fillna(0)
        time_series_filled['track_count'] = time_series_filled['track_count'].fillna(0)
        
        return time_series_filled

    def _filter_time_intervals(self, df):
        """Filter data based on time interval limit"""
        if self.filter_time_intervals != float('inf'):
            df = df[df["time_interval"] <= self.filter_time_intervals]
        return df

    def _create_summary_stats(self, duration_data):
        """Create summary statistics for duration data"""
        summary = (
            duration_data.groupby(self.treatment_or_rep)["duration_seconds"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )
        summary["se"] = summary["std"] / np.sqrt(summary["count"])
        summary["mean_upper_se"] = summary["mean"] + summary["se"]
        
        return summary

    def _create_time_series_summary(self, time_series_data):
        """Create summary statistics for time series data"""
        summary_time = (
            time_series_data.groupby([self.treatment_or_rep, "time_interval"])["sum_duration"]
            .agg(mean="sum", std="std", count="count")
            .reset_index()
        )
        summary_time["se"] = summary_time["std"] / np.sqrt(summary_time["count"])
        summary_time["mean_upper_se"] = summary_time["mean"] + summary_time["se"]
        summary_time["mean_lower_se"] = summary_time["mean"] - summary_time["se"]
        
        return summary_time

    def _creat_interval_sum(self, duration_data):
        interval_sum = duration_data.groupby([self.treatment_or_rep, "time_interval", "image_name"]).agg(
            sum_duration=("duration_seconds", "sum")
        ).reset_index()

        return interval_sum
    
    def save_new_df(self, data, name):
        output_csv = os.path.join(self.config['output_dir'], f"{name}.csv")
        create_output_dir(self.config['output_dir'])
        data.to_csv(output_csv, index=False)
    

    def box_plot_results(self,interval_sum):
        
        
        plot = (
            ggplot()
            + geom_boxplot(
                interval_sum,
                aes(x=self.treatment_or_rep, y="sum_duration", fill=self.treatment_or_rep),
                color="black",
                outlier_alpha=0.4
            )
            + geom_jitter(
                interval_sum,
                aes(x=self.treatment_or_rep, y="sum_duration"),
                color="black",
                width=0.2,
                alpha=0.8
            )
            + theme_classic()
            + labs(
                title="Duration of Mosquito Visits",
                x=" ",
                y="sum Duration (seconds)",
                fill=" "
            )
        )
        create_output_dir(self.plot_path)
        plot.save(os.path.join(self.plot_path, "sum_duration.png"))

    def time_plot_results(self, time_series_data, summary_time):
        """Create time series plot for duration"""
        plot = (
            ggplot(summary_time, aes(x="time_interval", y="mean", color=self.treatment_or_rep))
            + geom_point()
            + geom_line()
            + theme_classic()
            + labs(title="Average Duration Over Time", 
                   x=f"Time ({self.interval_unit})", 
                   y="Duration (seconds)", 
                   color=self.treatment_or_rep)
            + scale_x_continuous(
                limits=[0, time_series_data["time_interval"].max()], 
                breaks=range(0, int(time_series_data["time_interval"].max()) + 1, 
                           max(1, int(time_series_data["time_interval"].max() / 10)))
            )
        )
        
        if self.treatment_or_rep == "treatment":
            plot += geom_errorbar(summary_time, aes(x="time_interval", ymin="mean_lower_se", ymax="mean_upper_se"), width=0.2)
        
        create_output_dir(self.plot_path)
        plot.save(os.path.join(self.plot_path, "sum_duration_time.png"))

    def __call__(self):
        df = self._load_data()
        df = self._does_it_treatment_or_rep(df)
        
        # Sort data by image_idx to ensure proper frame order
        df = df.sort_values(["image_idx"])
        
        df = self._calculate_time_intervals(df)
        duration_data = self._calculate_duration(df)
        
        # Filter time intervals
        duration_data = self._filter_time_intervals(duration_data)
        
        # Create time series data
        time_series_data = self._aggregate_by_time_intervals(duration_data)
        
        # Create interval_summary

        interval_sum = self._creat_interval_sum(duration_data)
        summary_time = self._create_time_series_summary(time_series_data)
        
        # Save data
        self.save_new_df(data=duration_data, name="duration_data")
        self.save_new_df(data=interval_sum, name="interval_sum")
        self.save_new_df(data=time_series_data, name="duration_time_series")
        
        # Create plots
        self.box_plot_results(interval_sum)
        self.time_plot_results(time_series_data, summary_time)