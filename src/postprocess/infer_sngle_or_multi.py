import os
from typing import Dict
from src.inference.inference_pipeline import run_inference
from src.postprocess.sort_multi_videos import sort_and_merge_outputs
from src.utils.config_ops import export_config
from src.utils.config_ops import update_analyze_config

def inference_single_or_multi(config,conf_yaml_path, logger):
    export_config(conf_yaml_path, logger)
    source = config.get('images_dir', '')
    update_analyze = config.get('change_analyze_conf',False)
    flag = os.path.basename(source)
    
    if "." not in flag:  # It's likely a directory with multiple files
        vid_names = os.listdir(source)
        for vid_name in vid_names:
            export_name = vid_name.split(".")[0]
            config["images_dir"] = os.path.join(source, vid_name)
            config["csv_filename"] = f"{export_name}_results.csv"
            run_inference(config, logger)
    
    else:
        run_inference(config, logger)

    sort_and_merge_outputs(config, logger)
    if update_analyze:
        update_analyze_config(conf_yaml_path, logger)
   
