import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.config import load_config
import plotnine
from plotnine import ggplot, aes, geom_point, labs

class traj_explorer(object):
    def __init__(self, datapath):
        """Initialize with the path to the data."""
        self.data = pd.read_csv(datapath)
        data = self.data
        config = load_config(config_path)
        return data, config


    
    def plot_coords(data, output_plot="x_vs_y.png"):


        plot = (
            ggplot(data, aes(x='x', y='y', color='category')) +
            geom_point(size=3) +
            labs(title="Scatter Plot Example", x="X-axis Label", y="Y-axis Label")
        )
        plot.save(os.path.join(,output_plot))
        print(plot)

       

    def __call__(self, config):
        """Execute all steps based on the config dictionary."""
        datapath = config['data_path']
        plot_output_path = config['plot_output_path']
  
