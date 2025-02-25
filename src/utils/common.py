import yaml
import os
import mimetypes
import cv2
import pandas as pd
from typing import Union, List

def create_output_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



def is_video_or_image(path: str) -> str:
 
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type:
        if mime_type.startswith('image'):
            return 'image'
        elif mime_type.startswith('video'):
            return 'video'
    return None



def export_first_frame(input_path: Union[str, List[str]], output_dir: str,task: str):

    # Ensure input is a list
    if isinstance(input_path, str):
        video_paths = [input_path]
    else:
        video_paths = input_path
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_frames = []
    
    for video_path in video_paths:
        
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()
        
        if success:
            # Generate output path
            output_filename = os.path.splitext(os.path.basename(video_path))[0] + "_first_frame.jpg"
            output_file_path = os.path.join(output_dir,task, output_filename)
            cv2.imwrite(output_file_path, frame)
    
    return None


def find_image_for_heat_map(directory, target_string):
    # Find all files that contain the target string
    directory_list = os.listdir(directory)

    matches = [file for file in directory_list if target_string in file]
    jpg_matches = [file for file in matches if file.endswith('.jpg')]

    image_path = os.path.join(directory,jpg_matches[0])
    if image_path:
        return image_path
    else:
        return None
    
import os
import pandas as pd

def data_merger(directory, filename="results"):
    # List all files in the directory
    directory_list = os.listdir(directory)
    
    # Initialize an empty list to store data
    data = []
    
    # Variable to track the current maximum track_id
    current_max_track_id = 0
    
    # Loop through all files and append CSV data to the list
    for file in directory_list:
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            # Read CSV file
            df = pd.read_csv(file_path)
            
            df['track_id'] = df['track_id'] + current_max_track_id
            
            current_max_track_id = df['track_id'].max() + 1
            
            data.append(df)
    
    # Merge all data into one DataFrame
    merged_data = pd.concat(data, ignore_index=True)
    
    # Ensure the 'treatment' column is created
    if 'treatment' not in merged_data.columns:
        merged_data['treatment'] = merged_data['image_name'].apply(lambda x: x.split('_')[0])
    
    # Sort by 'track_id' to ensure numeric order
    merged_data = merged_data.sort_values(by='track_id')
    
    # Save the merged data to a new CSV file
    output_path = os.path.join(directory, f'{filename}.csv')
    merged_data.to_csv(output_path, index=False)
    
    return output_path



def update_yaml(config, yaml_path):
    """Update the YAML file with the latest changes in the config."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    # Update the 'input_csv' key with the new path
    data['input_csv'] = config['input_csv']
    
    # Write the updated data back to the YAML file
    with open(yaml_path, 'w') as file:
        yaml.safe_dump(data, file)

import pandas as pd
