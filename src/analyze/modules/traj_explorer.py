import os
import pandas as pd
from plotnine import ggplot, aes, geom_point, labs, theme_classic, theme, xlim,ylim
from src.utils.config import load_config
from src.utils.common import create_output_dir

class PlotXY:
    def __init__(self, config_path="config.json"):
       
        self.config = load_config(config_path)
        self.data_path = self.config['analyze']["csv_path"]
        self.plot_path = self.config["analyze"]["plots_dir"]
        self.results_output = self.config["analyze"]["csv_results_dir"]
        self.true_axis = self.config['analyze']['true_axis']
        self.id_OR_class = self.config['analyze']['id_OR_class']
        self.data = pd.read_csv(self.data_path)
        self.data[self.id_OR_class] = self.data[self.id_OR_class].astype('str')

    def plot_coords(self):
        
        create_output_dir(self.plot_path)
        for image_name in self.data["image_name"].unique():
            # Filter data for the current image
            image_data = self.data[self.data["image_name"] == image_name]
            
            # Create the plot
            plot = (
                ggplot(image_data, aes(x='x', y='y', color=self.id_OR_class)) +
                geom_point(size=3) +
                labs(title=f"X vs Y for {image_name}", x="X", y="Y") +
                theme_classic() +
                theme(legend_position="none")
            )
            
            # Adjust axis limits if true_axis is True
            if self.true_axis:
                plot += xlim(0, image_data['img_w'].iloc[0])
                plot +=ylim(image_data['img_h'].iloc[0], 0)
                
            
            # Define the output path and save the plot
            output_path = os.path.join(self.plot_path, f'{image_name}_x_y.png')
            try:
                plot.save(output_path)
                print(f"Plot saved at {output_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to save plot: {e}")

    def __call__(self):
     
        
        self.plot_coords()
        print("Processing and plotting complete!")
