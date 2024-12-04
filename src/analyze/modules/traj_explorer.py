import os
import pandas as pd
import plotnine as gg
from src.utils.config import load_config
from src.utils.common import create_output_dir

class PlotXY:
    def __init__(self, config_path="config.json"):
       
        self.config = load_config(config_path)
        self.data_file = os.path.join(self.config['output']['output_dir'], 'results.csv')
        self.plotpath = self.config['analyze']['output_dir']
        self.true_axis = self.config['analyze']['true_axis']
        self.id_OR_class = self.config['analyze']['id_OR_class']
        self.data = pd.read_csv(self.data_file)
        self.data[self.id_OR_class] = self.data[self.id_OR_class].astype('category')

    def plot_coords(self):
        
        create_output_dir(self.plotpath)

        plot = (
            gg.ggplot(self.data, gg.aes(x='x', y='y', color=self.data[self.id_OR_class])) +
            gg.geom_point(size=3) +
            gg.labs(title="X vs Y", x="X", y="Y")+
            gg.theme_classic()
        )
        if self.true_axis:
            plot = (
                plot +
                gg.xlim(0, self.data['img_w'].iloc[0]) +
                gg.ylim(0, self.data['img_h'].iloc[0])
            )
        output_path = os.path.join(self.plotpath, 'x_y.png')
        try:
            plot.save(output_path)
            print(f"Plot saved at {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save plot: {e}")

    def __call__(self):
     
        
        self.plot_coords()
        print("Processing and plotting complete!")
