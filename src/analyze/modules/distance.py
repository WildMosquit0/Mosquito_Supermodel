import os
import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes,
    geom_boxplot, theme_classic, labs,
    geom_jitter, geom_errorbar,
    geom_point, geom_line,
    scale_x_continuous
)
from src.utils.common import create_output_dir

class Distance:
    def __init__(self, config):
        self.config = config
        self.plot_path = f"{self.config['output_dir']}/plots"
        self.data_path = self.config["input_csv"]
        self.treatment_or_rep = self.config['plotxy']['treatment_or_rep']
        self.stat =  self.config.get("settings", {}).get("stat", "mean")
        self.fps = float(self.config.get("settings", {}).get("fps", 15))
        self.time_intervals = float(self.config.get("settings", {}).get("time_intervals", 1))
        self.interval_unit = self.config.get("settings", {}).get("interval_unit", "minutes")
        self.filter_time_intervals = self.config.get('settings', {}).get('filter_time_intervals', float('inf'))

    def _load_data(self):
        return pd.read_csv(self.data_path)

    def _does_it_treatment_or_rep(self, df):
        if self.treatment_or_rep not in df.columns:
            raise KeyError(f"Error: The column {self.treatment_or_rep} does not exist.")
        return df

    def _calculate_time_intervals(self, df):
        df["image_idx"] = pd.to_numeric(df["image_idx"], errors="coerce")
        df = df.dropna(subset=["image_idx"])
        conversion = 60 if self.interval_unit == "minutes" else 1
        length = self.time_intervals * conversion
        df["time_interval"] = ((df["image_idx"] / self.fps) / length).astype(int) * self.time_intervals
        return df

    def _calculate_distance_per_track(self, df):
        for col in ["track_id", "image_idx", "x", "y"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["track_id","image_idx","x","y"])
        df = df[["track_id","image_name","image_idx","x","y", self.treatment_or_rep,"time_interval"]]
        df = df.sort_values(["track_id","image_name","image_idx"])
        df["x_diff"] = df.groupby(["track_id","image_name"])["x"].diff()
        df["y_diff"] = df.groupby(["track_id","image_name"])["y"].diff()
        df["distance"] = np.sqrt(df["x_diff"]**2 + df["y_diff"]**2)
        df = df.dropna(subset=["distance"])
        track_dist = (
            df.groupby(["track_id","image_name",self.treatment_or_rep])
              .agg(total_distance=("distance","sum"),
                   frame_count=("image_idx","count"))
              .reset_index()
        )
        return df, track_dist

    def _sum_distance_by_time_interval(self, df):
        """Sum distances and ensure full grid of intervals × replicates × treatments."""
        tid = (
            df.groupby(["time_interval","image_name",self.treatment_or_rep])
              .agg(total_distance=("distance","sum"),
                   track_count=("track_id","nunique"))
              .reset_index()
        )
        raw_reps = (
            pd.read_csv(self.data_path)[["image_name", self.treatment_or_rep]]
            .drop_duplicates()
        )
        max_int = int(tid["time_interval"].max())
        all_ints = np.arange(0, max_int + self.time_intervals, self.time_intervals)
        grid = [
            {
                "time_interval": t,
                "image_name": r["image_name"],
                self.treatment_or_rep: r[self.treatment_or_rep]
            }
            for _, r in raw_reps.iterrows()
            for t in all_ints
        ]
        full = pd.DataFrame(grid)
        filled = (
            full.merge(tid,
                       on=["time_interval","image_name",self.treatment_or_rep],
                       how="left")
                .fillna({"total_distance": 0, "track_count": 0})
        )
        return filled

    def _calculate_bar_summary(self, ti_dist):
        summary = ti_dist.groupby(self.treatment_or_rep)["total_distance"].agg(
            **{ self.stat: self.stat, "std": "std", "count": "count" }
        ).reset_index()
        summary["se"] = summary["std"] / np.sqrt(summary["count"])
        summary[f"{self.stat}_upper_se"] = summary[self.stat] + summary["se"]
        summary[f"{self.stat}_lower_se"] = summary[self.stat] - summary["se"]
        return summary

    def _calculate_time_summary(self, ti_dist):
        summary_time = ti_dist.groupby([self.treatment_or_rep,"time_interval"])["total_distance"].agg(
            **{ self.stat: self.stat, "std": "std", "count": "count" }
        ).reset_index()
        summary_time["se"] = summary_time["std"] / np.sqrt(summary_time["count"])
        summary_time[f"{self.stat}_upper_se"] = summary_time[self.stat] + summary_time["se"]
        summary_time[f"{self.stat}_lower_se"] = summary_time[self.stat] - summary_time["se"]
        return summary_time

    def _filter_time_intervals(self, df):
        if self.filter_time_intervals != float('inf'):
            return df[df["time_interval"] <= self.filter_time_intervals]
        return df

    def save_new_df(self, data, name):
        path = os.path.join(self.config['output_dir'], f"{name}.csv")
        create_output_dir(self.config['output_dir'])
        data.to_csv(path, index=False)

    def box_plot_results(self, ti_dist):
        p = (
            ggplot()
            + geom_boxplot(
                ti_dist,
                aes(x=self.treatment_or_rep, y="total_distance", fill=self.treatment_or_rep),
                color="black", outlier_alpha=0.4
            )
            + geom_jitter(
                ti_dist,
                aes(x=self.treatment_or_rep, y="total_distance"),
                color="black", width=0.2, alpha=0.8
            )
            + theme_classic()
            + labs(
                title=f"Distance by {self.stat.title()}",
                x=" ",
                y=f"Distance ({self.stat})"
            )
        )
        create_output_dir(self.plot_path)
        p.save(os.path.join(self.plot_path, f"distance_bar_{self.stat}.png"))

    def time_plot_results(self, ti_dist, summary_time):
        p = (
            ggplot(summary_time, aes(x="time_interval", y=self.stat, color=self.treatment_or_rep))
            + geom_point()
            + geom_line()
            + theme_classic()
            + labs(
                title=f"Distance over Time ({self.stat.title()})",
                x=f"Time ({self.interval_unit})",
                y=f"Distance ({self.stat})"
            )
            + scale_x_continuous(
                limits=[0, ti_dist["time_interval"].max()],
                breaks=range(
                    0,
                    int(ti_dist["time_interval"].max()) + 1,
                    max(1, int(ti_dist["time_interval"].max() / 10))
                )
            )
        )
        if self.treatment_or_rep == "treatment":
            p += geom_errorbar(
                summary_time,
                aes(x="time_interval",
                    ymin=f"{self.stat}_lower_se",
                    ymax=f"{self.stat}_upper_se"),
                width=0.2
            )
        create_output_dir(self.plot_path)
        p.save(os.path.join(self.plot_path, f"distance_time_{self.stat}.png"))

    def __call__(self):
        # 1. load & verify
        df = self._load_data()
        df = self._does_it_treatment_or_rep(df)
        df = df.sort_values("image_idx")

        # 2. compute time intervals (must come before distance)
        df = self._calculate_time_intervals(df)

        # 3. per‐track distances
        df_dist, _ = self._calculate_distance_per_track(df)
        df_dist = self._filter_time_intervals(df_dist)

        # 4. sum by time interval with full grid
        ti_dist = self._sum_distance_by_time_interval(df_dist)

        # 5. build summaries
        bar_summary  = self._calculate_bar_summary(ti_dist)
        time_summary = self._calculate_time_summary(ti_dist)

        # 6. save only the tables your plots consume
        self.save_new_df(ti_dist,      "distance_by_time_interval")
        self.save_new_df(time_summary, f"{self.stat}_summary_over_time")

        # 7. render plots
        self.box_plot_results(ti_dist)
        self.time_plot_results(ti_dist, time_summary)
