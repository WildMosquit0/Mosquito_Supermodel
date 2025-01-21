from src.analyze.modules.traj_explorer import PlotXY
from src.analyze.modules.average_visits import AverageVisits
from src.analyze.modules.duration import Duration

def run_analysis(config):
    explorer = PlotXY(config)
    explorer()
    average_visits = AverageVisits(config)
    average_visits()
    duration = Duration(config)
    duration()