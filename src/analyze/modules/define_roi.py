import cv2
import pandas as pd
import os
import sys

# Get the current directory
current_dir = os.path.abspath(os.getcwd())

# Move up until "src" is in the directory name
while True:
    if 'src' in os.listdir(current_dir):
        break
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:  # Reached the root directory
        raise FileNotFoundError("Directory 'src' not found.")
    current_dir = parent_dir

# Append to sys.path
sys.path.append(current_dir)
print(f"Added to sys.path: {current_dir}")
from src.utils.config import load_config
from src.utils.common import create_output_dir

class ROIDefiner:
    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)
        self.folder_path = self.config["define_roi"]["file_path"]
        self.csv_path = self.config["define_roi"]["csv_path"]
        self.output_dir = self.config["define_roi"]["output_dir"]
        self.get_inner_roi = self.config["define_roi"]["get_inner_roi"]
        self.choose_frame_for_video = self.config["define_roi"]["choose_farme_for_video"]
        self.roi = None
        self.current_frame = None

    def get_video_files(self):
        """Get a list of video files in the folder."""
        return [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))
        ]

    def load_video(self, filepath):
        """Load a video file."""
        self.video = cv2.VideoCapture(filepath)
        if not self.video.isOpened():
            raise ValueError(f"Unable to open the video file: {filepath}")
        return filepath

    def choose_frame(self, frame_number=0):
        """Select a specific frame from the video."""
        if self.video is None:
            raise ValueError("No video file loaded. Cannot choose a frame.")
        
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video.read()
        if ret:
            self.current_frame = frame
        else:
            raise ValueError(f"Cannot read frame {frame_number} from video.")

    def select_roi(self, image):
        """Allow the user to define a region of interest (ROI) on an image."""
        self.roi = cv2.selectROI("Select ROI", image, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        return self.roi

    def define_roi_for_video(self, filepath):
        """Define ROI for a specific video."""
        self.load_video(filepath)
        frame_number = self.choose_frame_for_video
        self.choose_frame(frame_number)
        print(f"Displaying frame {frame_number} from video: {filepath}")
        self.roi = self.select_roi(self.current_frame)
        return self.roi

    def load_csv(self):
        """Load the CSV file."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        self.object_data = pd.read_csv(self.csv_path)
        print(f"Loaded CSV with {len(self.object_data)} rows.")
        return self.object_data

    def filter_objects_in_roi(self, image_name):
        """Filter objects in the ROI for a specific video."""
        if self.roi is None:
            raise ValueError("ROI not defined. Please define an ROI before filtering objects.")
        
        x, y, w, h = self.roi
        x_start, y_start = x, y
        x_end, y_end = x + w, y + h

        filtered_data = self.object_data[
            (self.object_data['image_name'] == image_name) &
            (
                (self.object_data['x'] >= x_start) & 
                (self.object_data['y'] >= y_start) & 
                (self.object_data['x'] <= x_end) & 
                (self.object_data['y'] <= y_end)
            )
        ]
        print(f"Filtered {len(filtered_data)} objects inside the ROI for image_name: {image_name}.")
        return filtered_data

    def save_filtered_data(self, filtered_data, filename="roi_results.csv"):
        """Save the filtered data to a CSV file."""
        create_output_dir(self.output_dir)
        output_path = os.path.join(self.output_dir, filename)
        filtered_data.to_csv(output_path, index=False)
        print(f"Filtered data saved to {output_path}.")

    def __call__(self):
        """Process all videos in the folder."""
        video_files = self.get_video_files()
        all_filtered_data = pd.DataFrame()

        for video_file in video_files:
            image_name = os.path.basename(video_file)
            print(f"Processing video: {image_name}")
            roi = self.define_roi_for_video(video_file)
            print(f"ROI for {image_name}: {roi}")

            # Load CSV and filter objects
            csv_data = self.load_csv()
            filtered_data = self.filter_objects_in_roi(image_name)
            all_filtered_data = pd.concat([all_filtered_data, filtered_data])

        # Save all results to a single file
        self.save_filtered_data(all_filtered_data, filename="roi_results.csv")

# Instantiate and run the ROIDefiner class
config_path = "./config.json"
roi_definer = ROIDefiner(config_path)
roi_definer()
