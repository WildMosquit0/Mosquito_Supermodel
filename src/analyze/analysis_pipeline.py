from src.analyze.modules.traj_explorer import PlotXY
from src.analyze.modules.average_visits import AverageVisits
from src.analyze.modules.duration import Duration
from src.analyze.modules.heatmap import Heatmap
from src.utils.common import data_merger
from src.utils.common import update_yaml

import os

def run_analysis(config,conf_yaml_path):

    if os.path.isdir(config['input_csv']):
            merged_data_path = data_merger(config['input_csv'])
            config['input_csv'] = merged_data_path
            update_yaml(config, conf_yaml_path)
            
    if config['task']['plotxy']:
        explorer = PlotXY(config)
        explorer()
    if config['task']['average_visits']:
        average_visits = AverageVisits(config)
        average_visits()
    if config['task']['duration']:
        duration = Duration(config)
        duration()
    if config['task']['heatmap']:
        heatmap = Heatmap(config)
        heatmap()