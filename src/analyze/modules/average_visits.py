import os
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_col, theme_classic, labs, geom_point, geom_line, geom_jitter, geom_errorbar, scale_x_continuous
from src.utils.common import create_output_dir

class AverageVisits:
    def __init__(self, config):
        self.config = config
        self.plot_path = f"{self.config['output_dir']}/plots"
        self.time_intervals = float(self.config["average_visits"]["time_intervals"])
        self.fps = float(self.config["average_visits"]["fps"])
        self.filter_time_intervals = self.config['average_visits'].get('filter_time_intervals', "nan")
        self.data_path = self.config["input_csv"]
        self.teratment_or_rep = self.config['plotxy']['teratment_or_rep']

    def _load_data(self):
        return pd.read_csv(self.data_path)

    def _does_it_teratment_or_rep(self, df):
        if self.teratment_or_rep not in df.columns:
            raise KeyError(f"Error: The column {self.teratment_or_rep} does not exist in the DataFrame.")
        df['teratment_or_rep'] = df[self.teratment_or_rep].astype('str')
        return df

    def fill_na_time_intervals(self, df):
        # Determine the maximum time interval
        max_interval = df["time_interval"].max()
        # Create a DataFrame of all possible time intervals
        all_intervals = pd.DataFrame({"time_interval": range(0, max_interval + 1)})
        # Get the unique treatments from the column specified by self.teratment_or_rep
        unique_treatments = df[self.teratment_or_rep].unique()
        
        # Create a complete grid of all time_interval and treatment combinations
        complete_grid = (
            pd.MultiIndex.from_product(
                [all_intervals["time_interval"], unique_treatments],
                names=["time_interval", self.teratment_or_rep]
            )
            .to_frame(index=False)
        )
        
        # Merge the complete grid with the original DataFrame using a left join.
        # This preserves duplicate rows in the original data and fills in missing combinations.
        df_filled = complete_grid.merge(df, on=["time_interval", self.teratment_or_rep], how="left")
        
        # Fill missing values with 0
        df = df_filled.fillna(0)
        
        return df


    def _calculate_time_intervals(self, df):
        df["image_idx"] = pd.to_numeric(df["image_idx"], errors="coerce")
        df = df.dropna(subset=["image_idx"])
        divisor = self.fps * self.time_intervals
        if divisor == 0:
            raise ValueError("Divisor (fps * time_intervals) cannot be zero.")
        df["time_interval"] = (df["image_idx"] / divisor).astype(int)
        return df

    def _aggregate_trajectories(self, df):
        df["trajectory_count"] = df.groupby(["image_idx", "image_name", "teratment_or_rep", "time_interval"])["box_idx"].transform("nunique")
        df["time"] = df["image_idx"] / self.fps / 60

        average_visits = (
            df.groupby(['time_interval', "image_name"], as_index=False)
            .agg(mean_trajectory_count=('trajectory_count', 'mean'))
        )
        average_visits["treatment"] = average_visits["image_name"].str.split("_").str[0]
        return self.fill_na_time_intervals(average_visits)

    def _filter_time_intervals(self, df):
        if self.filter_time_intervals != "nan":
            df = df[df["time_interval"] <= self.filter_time_intervals]
        return df

    def create_summary_stats_for_graphs(self, average_visits):
        summary_bar = (
            average_visits.groupby(self.teratment_or_rep)["mean_trajectory_count"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )
        summary_bar["se"] = summary_bar["std"] / np.sqrt(summary_bar["count"])
        summary_bar["mean_upper_se"] = summary_bar["mean"] + summary_bar["se"]

        summary_time = (
            average_visits.groupby([self.teratment_or_rep, "time_interval"])["mean_trajectory_count"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )
        summary_time["se"] = summary_time["std"] / np.sqrt(summary_time["count"])
        summary_time["mean_upper_se"] = summary_time["mean"] + summary_time["se"]
        summary_time["mean_lower_se"] = summary_time["mean"] - summary_time["se"]

        return summary_bar, summary_time


    def save_new_df(self, data, name):
        output_csv = os.path.join(self.config['output_dir'], f"{name}.csv")
        create_output_dir(self.config['output_dir'])
        data.to_csv(output_csv, index=False)

    def bar_plot_results(self, average_visits, summary_bar):
        plot = (
            ggplot()
            + geom_col(summary_bar, aes(x=self.teratment_or_rep, y="mean", fill=self.teratment_or_rep), color="black")
            + geom_jitter(average_visits, aes(x=self.teratment_or_rep, y="mean_trajectory_count"), color="black", width=0.2, alpha=0.8)
            + theme_classic()
            + labs(title=" ", x=" ", y="Mean Trajectory Count", fill=" ")
        )
        if self.teratment_or_rep == "treatment":
            plot += geom_errorbar(summary_bar, aes(x=self.teratment_or_rep, ymin="mean", ymax="mean_upper_se"), width=0.2)
        create_output_dir(self.plot_path)
        plot.save(os.path.join(self.plot_path, "average_visits.png"))

    def time_plot_results(self, average_visits, summary_time):
        plot = (
            ggplot(summary_time, aes(x="time_interval", y="mean", color=self.teratment_or_rep))
            + geom_point()
            + geom_line()
            + theme_classic()
            + labs(title=" ", x="Time (minutes)", y="Trajectory's number", color=" ")
            + scale_x_continuous(limits=[0, average_visits["time_interval"].max()], breaks=range(0, average_visits["time_interval"].max() + 1, 3))
        )
        if self.teratment_or_rep == "treatment":
            plot += geom_errorbar(summary_time, aes(x="time_interval", ymin="mean_lower_se", ymax="mean_upper_se"), width=0.2)
        create_output_dir(self.plot_path)
        plot.save(os.path.join(self.plot_path, "average_visits_time.png"))

    def __call__(self):
        df = self._load_data()
        df = self._does_it_teratment_or_rep(df)
        df = self._calculate_time_intervals(df)
        average_visits = self._aggregate_trajectories(df)
        average_visits = self._filter_time_intervals(average_visits)
        summary_bar, summary_time = self.create_summary_stats_for_graphs(average_visits)
        self.save_new_df(data=average_visits, name="average_visits")
        self.bar_plot_results(average_visits, summary_bar)
        self.time_plot_results(average_visits, summary_time)