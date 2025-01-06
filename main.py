from src.inference.inference_pipeline import main as run_inference
from src.analyze.modules.traj_explorer import PlotXY
from src.analyze.modules.average_visits import AverageVisits
from src.analyze.modules.duration import Duration
from src.analyze.modules.define_roi import ROIDefiner 

if __name__ == "__main__":
    # Use the relative path to your config file
    config_path = "./config.json"

    # ! Step 1: Run the following line to predict/track
    #run_inference(config_path=config_path)
    
    # ! Step 2: Run the following line to explore and analyze the model's trajectories
    define_roi_main = ROIDefiner(config_path)
    explorer = PlotXY(config_path)
    average_visits = AverageVisits(config_path)
    #duration = Duration(config_path)
    
    
    #explorer()
    #average_visits()
    #duration()
    define_roi_main()
    
    # ** to define ROI for video, uncomment the following line **
    #! remember to adjust the 'csv path' in the config file to /path/to/your/roi_results.csv
    
    