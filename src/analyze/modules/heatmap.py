import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.utils.common import create_output_dir
from src.utils.common import find_image_for_heat_map


class Heatmap:
    def __init__(self, config):
        self.config = config
        self.data_path = self.config["input_csv"]
        self.plot_path = os.path.join(self.config["output_dir"], "plots")
        self.frame_path =  self.config["heatmap"]["image_path"]
        self.grid_size = self.config.get("grid_size", 70)
        self.min_count = self.config.get("min_count", 10)
        self.cmap = self.config.get("cmap", "plasma")
        
         
        
    def plot_heatmap(self):
        name =  os.path.basename(self.data_path).split('.')[0]
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
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{name} Heatmap")
        
        output_path = os.path.join(self.plot_path, f"{name} Heatmap.png")
        try:
            plt.savefig(output_path)
            print(f"Plot saved at {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save plot: {e}")
        
        plt.show()

    def __call__(self):
        
        source = self.data_path
        flag = os.path.basename(source)
        if flag.find(".") <= 0:
            csv_names = os.listdir(source)
            for data in csv_names:
                    if data.endswith(".csv"):

                        self.data_path = os.path.join(source, data)
                        self.data = pd.read_csv(self.data_path) 
                        image_name = self.data['image_name'][0]
                        image_path = find_image_for_heat_map(source,image_name)
                        self.frame_path = image_path
                        self.plot_heatmap()
        else:
            self.data = pd.read_csv(self.data_path)
            self.plot_heatmap()
        
