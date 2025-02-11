from src.analyze.modules.traj_explorer import PlotXY
from src.analyze.modules.average_visits import AverageVisits
from src.analyze.modules.duration import Duration
from src.analyze.modules.heatmap import Heatmap

def run_analysis(config):
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