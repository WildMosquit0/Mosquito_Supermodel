from src.inference.inference_pipeline import main as run_inference

if __name__ == "__main__":
    # Use the relative path to your config file
    config_path = "./config.json"

    # Call the inference function with the config path
    run_inference(config_path=config_path)
