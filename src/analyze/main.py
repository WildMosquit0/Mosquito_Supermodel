from modules.x_y_exploration import x_y_exploration

def analyze_main(config_path="src/analyze/config.yaml"):
    """Main function to run the analysis part."""
    x_y_exploration.main(config_path)

if __name__ == "__main__":
    analyze_main()
