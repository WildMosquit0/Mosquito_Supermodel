import os
import pandas as pd
from plotnine import ggplot, aes, geom_point, labs, theme_classic, theme, xlim,ylim
from src.utils.common import create_output_dir

class PlotXY:
    def __init__(self, config):
        self.config = config
        self.data_path = self.config["input_csv"]
        self.plot_path = f"{self.config['output_dir']}/plots"

        self.true_axis = self.config['plotxy']['true_axis']
        self.id_OR_class = self.config['plotxy']['id_OR_class']
        self.data = pd.read_csv(self.data_path)
        self.data[self.id_OR_class] = self.data[self.id_OR_class].astype('str')

    def plot_coords(self):
        
        create_output_dir(self.plot_path)
        for image_name in self.data["image_name"].unique():
            image_data = self.data[self.data["image_name"] == image_name]
            
            plot = (
                ggplot(image_data, aes(x='x', y='y', color=self.id_OR_class)) +
                geom_point(size=1) +
                labs(title=f"X vs Y for {image_name}", x="X", y="Y") +
                theme_classic() +
                theme(legend_position="none")
            )
            
            if self.true_axis:
                plot += xlim(0, image_data['img_w'].iloc[0])
                plot +=ylim(image_data['img_h'].iloc[0], 0)
                
            
            output_path = os.path.join(self.plot_path, f'{image_name}_x_y.png')
            try:
                plot.save(output_path)
                print(f"Plot saved at {output_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to save plot: {e}")

    def __call__(self):
        self.plot_coords()
        print("Processing and plotting complete!")
