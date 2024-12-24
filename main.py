from src.inference.inference_pipeline import main as run_inference
from src.analyze.modules.traj_explorer import PlotXY
from src.analyze.modules.average_visits import AverageVisits
from src.analyze.modules.duration import Duration


if __name__ == "__main__":
    # Use the relative path to your config file
    config_path = "./config.json"

    # Step 1: Run inference pipeline (if applicable)
    #run_inference(config_path=config_path)
    
    # Step 2: Initialize and execute PlotXY for data visualization
    explorer = PlotXY(config_path)
    average_visits = AverageVisits(config_path)
    duration = Duration(config_path)
    
    
    explorer()
    average_visits()
    duration()