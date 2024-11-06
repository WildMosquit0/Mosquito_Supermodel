import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

class x_y_exploration(object):
    def __init__(self, datapath="path from config"):
        self.data = pd.read_csv(datapath)

    def get_x_y_coordinates(self):
        """Extract x1, y1, x2, and y2 coordinates from the dataset."""
        x1 = self.data['x1'].tolist()
        y1 = self.data['y1'].tolist()
        x2 = self.data['x2'].tolist()
        y2 = self.data['y2'].tolist()
        return x1, x2, y1, y2

    def get_id(self):
        """Extract IDs from the dataset."""
        id = self.data['track_id'].tolist()
        return id

    def get_classes(self):
        """Retrieve the class labels."""
        classes = self.data['label'].tolist()
        return classes

    def calculate_average(self, x1, y1, x2, y2):
        """Calculate the midpoint average of x and y coordinates."""
        x = [(x1[i] + x2[i]) / 2 for i in range(len(x1))]
        y = [(y1[i] + y2[i]) / 2 for i in range(len(y1))]
        return x, y

    def save_vectors(self, x, y, id, classes, output_path):
        """Save the calculated x and y vectors to a CSV file."""
        vectors_df = pd.DataFrame({'x': x, 'y': y, 'id': id, 'classes': classes})
        vectors_df.to_csv(output_path, index=False)

    def plot_coords(self, x, y, id=None, classes=None, output_path="plot.png", xlim=None, ylim=None):
        """Plot and save coordinates as a scatter plot with optional limits and unique colors for id or classes."""
        
        # Check if both id and classes are provided, and raise an error if so
        if id is not None and classes is not None:
            raise ValueError("You can choose only 'classes' or 'id', not both.")
        
        # Determine the coloring basis and set legend label accordingly
        if classes is not None:
            color_basis = classes
            legend_label = "Class"
        elif id is not None:
            color_basis = id
            legend_label = "ID"
        else:
            color_basis = None
            legend_label = None

        # Plot with color mapping if color_basis is provided
        if color_basis is not None:
            unique_values = list(set(color_basis))
            color_map = {val: idx for idx, val in enumerate(unique_values)}
            colors = [color_map[val] for val in color_basis]

            # Scatter plot with colors and a dynamic legend label
            scatter = plt.scatter(x, y, c=colors, cmap='viridis', label=legend_label)
            colorbar = plt.colorbar(scatter, ticks=range(len(unique_values)), label=legend_label)
            scatter.set_clim(-0.5, len(unique_values) - 0.5)
        else:
            # Plot without color mapping if neither id nor classes provided
            plt.scatter(x, y)

        plt.xlabel("X")
        plt.ylabel("Y")

        # Apply axis limits if provided
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

        plt.savefig(output_path)
        plt.close()


    @classmethod
    def main(cls, config_path="config.yaml"):
        """Main function to execute all steps based on the config file."""
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        
        datapath = config['data_path']
        plot_output_path = config['plot_output_path']
        vector_output_path = config['vector_output_path']
        xlim = config.get('xlim', None)
        ylim = config.get('ylim', None)

        exploration = cls(datapath)

        x1, x2, y1, y2 = exploration.get_x_y_coordinates()
        mid_x, mid_y = exploration.calculate_average(x1, y1, x2, y2)

        ids = exploration.get_id() if 'track_id' in exploration.data.columns else None
        classes = exploration.get_classes() if 'label' in exploration.data.columns else None

        exploration.plot_coords(mid_x, mid_y, output_path=plot_output_path, xlim=xlim, ylim=ylim)
        exploration.save_vectors(mid_x, mid_y, ids, classes, vector_output_path)
