from src.analyze.modules.traj_explorer import PlotXY
from src.analyze.modules.average_visits import AverageVisits
from src.analyze.modules.duration import Duration
from src.analyze.modules.distance import Distance
from src.analyze.modules.heatmap import Heatmap
from src.utils.common import data_merger
from src.utils.common import update_yaml

import os

def run_analysis(config,conf_yaml_path):

    if os.path.isdir(config['input_csv']):
            merged_data_path = data_merger(config['input_csv'])
            config['input_csv'] = merged_data_path
            update_yaml(config, conf_yaml_path,'input_csv')
            
       # Run enabled tasks
    if config['task']['average_visits']:
        print("Running Average Visits analysis...")
        avg_visits = AverageVisits(config)
        avg_visits()
    
    if config['task']['duration']:
        print("Running Duration analysis...")
        duration = Duration(config)
        duration()
    
    if config['task'].get('distance', False):
        print("Running Distance Traveled analysis...")
        distance = Distance(config)
        distance()
    
    if config['task']['heatmap']:
        print("Running Heatmap analysis...")
        heatmap = Heatmap(config)
        heatmap()
    
    if config['task']['plotxy']:
        print("Running PlotXY analysis...")
        plotxy = PlotXY(config)
        plotxy()
    