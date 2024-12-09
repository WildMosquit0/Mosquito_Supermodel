import cv2
import pandas as pd
import os
import sys

sys.path.append(r'C:\Users\bohbot-lab\git\Mosquito_Supermodel_backup')
from src.utils.config import load_config
from src.utils.common import create_output_dir

class ROIDefiner:
    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)
        self.filepath = self.config["define_roi"]["file_path"]
        self.csv_path = self.config["define_roi"]["csv_path"]
        self.output_dir = self.config["define_roi"]["output_dir"]
        self.get_inner_roi = self.config["define_roi"]["get_inner_roi"]
        self.choose_frame_for_video = self.config["define_roi"]["choose_farme_for_video"]
        self.roi = None
        self.image = None
        self.video = None
        self.current_frame = None

    def get_file_type(self):
        if self.filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            return 'image'
        elif self.filepath.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            return 'video'
        else:
            raise ValueError("Unsupported file type. Please provide an image or video.")

    def load_file(self):
        file_type = self.get_file_type()
        if file_type == 'image':
            self.image = cv2.imread(self.filepath)
            if self.image is None:
                raise ValueError("Unable to read the image file.")
        elif file_type == 'video':
            self.video = cv2.VideoCapture(self.filepath)
            if not self.video.isOpened():
                raise ValueError("Unable to open the video file.")

    def choose_frame(self, frame_number=0):
        if self.video is None:
            raise ValueError("No video file loaded. Cannot choose a frame.")
        
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video.read()
        if ret:
            self.current_frame = frame
        else:
            raise ValueError(f"Cannot read frame {frame_number} from video.")

    def select_roi(self, image):
        self.roi = cv2.selectROI("Select ROI", image, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        return self.roi

    def define_roi(self):
        """Load the file, display it, and allow the user to define the ROI."""
        self.load_file()
        if self.image is not None:
            print("Loaded an image.")
            print("Please select ROI.")
            self.roi = self.select_roi(self.image)
        elif self.video is not None:
            print("Loaded a video.")
            frame_number = self.choose_frame_for_video
            self.choose_frame(frame_number)
            print(f"Displaying frame {frame_number}.")
            print("Please select ROI.")
            self.roi = self.select_roi(self.current_frame)

    def load_csv(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        self.object_data = pd.read_csv(self.csv_path)
        print(f"Loaded CSV with {len(self.object_data)} rows.")
        return self.object_data

    def filter_objects_in_roi(self):
        if not hasattr(self, 'get_inner_roi'):
            raise AttributeError("The attribute 'get_inner_roi' is not defined. Please set it before calling this method.")
        
        if self.roi is None:
            raise ValueError("ROI not defined. Please define an ROI before filtering objects.")
        
        x, y, w, h = self.roi
        x_start, y_start = x, y
        x_end, y_end = x + w, y + h

        if self.get_inner_roi:
            # Filter objects inside the ROI
            filtered_data = self.object_data[
                (self.object_data['x'] >= x_start) & 
                (self.object_data['y'] >= y_start) & 
                (self.object_data['x'] <= x_end) & 
                (self.object_data['y'] <= y_end)
            ]
            print(f"Filtered {len(filtered_data)} objects inside the ROI.")
        else:
            # Filter objects outside the ROI
            filtered_data = self.object_data[
                ~(
                    (self.object_data['x'] >= x_start) & 
                    (self.object_data['y'] >= y_start) & 
                    (self.object_data['x'] <= x_end) & 
                    (self.object_data['y'] <= y_end)
                )
            ]
            print(f"Filtered {len(filtered_data)} objects outside the ROI.")
        
        return filtered_data

    def save_filtered_data(self, filtered_data, filename="roi_results.csv"):
        create_output_dir(self.output_dir)
        output_path = os.path.join(self.output_dir, filename)
        filtered_data.to_csv(output_path, index=False)
        print(f"Filtered data saved to {output_path}.")

    def __call__(self):
        # Define the ROI
        self.define_roi()

        # Load CSV
        csv_data = self.load_csv()

        # Filter objects in ROI
        objects_in_roi = self.filter_objects_in_roi()
        self.save_filtered_data(objects_in_roi)

# Instantiate the ROIDefiner class
config_path = "./config.json"
roi_definer = ROIDefiner(config_path)

# Trigger the __call__ method
roi_definer()
