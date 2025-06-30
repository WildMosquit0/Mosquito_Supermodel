import os
import shutil
import pandas as pd
from typing import Dict

def sort_and_merge_outputs(config: Dict):
    output_dir = config["output_dir"]

    # Create subfolders
    videos_dir = os.path.join(output_dir, "videos")
    frames_dir = os.path.join(output_dir, "frames")
    csvs_dir = os.path.join(output_dir, "csvs")

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(csvs_dir, exist_ok=True)

    # Detect first-level output subdir (e.g., predict, track, slice)
    subdirs = [d for d in os.listdir(output_dir)
               if os.path.isdir(os.path.join(output_dir, d)) and d not in ['videos', 'frames', 'csvs']]

    if not subdirs:
        print("No inference output subfolder found.")
        return

    data_dir = os.path.join(output_dir, subdirs[0])

    # Move files to respective folders
    for fname in os.listdir(data_dir):
        full_path = os.path.join(data_dir, fname)
        if fname.endswith(".avi"):
            shutil.move(full_path, os.path.join(videos_dir, fname))
        elif fname.endswith(".jpg"):
            shutil.move(full_path, os.path.join(frames_dir, fname))
        elif fname.endswith(".csv") and fname != "results.csv":
            shutil.move(full_path, os.path.join(csvs_dir, fname))

    # Merge CSVs from csvs_dir while updating track_id
    merged_data = []
    current_max_track_id = 0

    for file in sorted(os.listdir(csvs_dir)):
        if file.endswith('.csv'):
            file_path = os.path.join(csvs_dir, file)
            df = pd.read_csv(file_path)

            if 'track_id' in df.columns:
                df['track_id'] = df['track_id'] + current_max_track_id
                current_max_track_id = df['track_id'].max() + 1
            else:
                print(f"Warning: 'track_id' not found in {file}, skipping track ID offset.")

            if 'image_name' in df.columns and 'treatment' not in df.columns:
                df['treatment'] = df['image_name'].apply(lambda x: str(x).split('_')[0])

            merged_data.append(df)

    if not merged_data:
        print("No CSVs to merge.")
        return

    final_df = pd.concat(merged_data, ignore_index=True).sort_values(by='track_id')
    output_path = os.path.join(output_dir, "results.csv")
    final_df.to_csv(output_path, index=False)

    print(f"Merged results saved to {output_path}")
    return output_path