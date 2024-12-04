from src.inference.inference_pipeline import main as run_inference
from src.analyze.modules.traj_explorer import PlotXY

if __name__ == "__main__":
    # Use the relative path to your config file
    config_path = "./config.json"

    # Step 1: Run inference pipeline (if applicable)
    run_inference(config_path=config_path)
    
    # Step 2: Initialize and execute PlotXY for data visualization
    explorer = PlotXY(config_path)
    explorer()
