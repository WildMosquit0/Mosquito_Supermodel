import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.utils.common import create_output_dir

class Heatmap:
    def __init__(self, config):
        self.config = config
        self.data_path = self.config["input_csv"]
        self.plot_path = os.path.join(self.config["output_dir"], "plots")
        self.frame_path =  self.config["heatmap"]["image_path"]
        self.grid_size = self.config.get("grid_size", 200)
        self.min_count = self.config.get("min_count", 4)
        self.cmap = self.config.get("cmap", "plasma")
        
        self.data = pd.read_csv(self.data_path)

    def plot_heatmap(self):
        
        create_output_dir(self.plot_path)

        fig, ax = plt.subplots(figsize=(10, 8))
        frame = plt.imread(self.frame_path)
    
        
        plt.hexbin(
            self.data["x"], self.data["y"], 
            gridsize=self.grid_size, cmap=self.cmap, mincnt=self.min_count
        )
        ax.imshow(frame)
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Heatmap Over Video Frame")
        
        output_path = os.path.join(self.plot_path, "heatmap.png")
        try:
            plt.savefig(output_path)
            print(f"Plot saved at {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save plot: {e}")
        
        plt.show()

    def __call__(self):
        self.plot_heatmap()
        print("Processing and plotting complete!")
