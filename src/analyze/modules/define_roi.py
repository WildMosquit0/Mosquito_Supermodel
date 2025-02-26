import cv2
import pandas as pd
import os
import sys
import yaml

# Get the current directory and ensure "src" is in the path
current_dir = os.path.abspath(os.getcwd())

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

from src.utils.common import create_output_dir


def load_config_yaml(config_path):
    """Load the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class ROIDefiner:
    def __init__(self, config_path="configs/roi.yaml"):
        """Initialize the ROI Definer using a YAML config."""
        self.config = load_config_yaml(config_path)

        # Updated to match the revised YAML structure:
        self.folder_path = self.config["define_roi"]["folder_path"]
        self.csv_path = self.config["define_roi"]["csv_path"]
        self.output_dir = self.config["define_roi"]["output_dir"]
        self.get_inner_roi = self.config["define_roi"]["get_inner_roi"]

        # Key for which frame to pick from video
        self.choose_frame_for_video = self.config["define_roi"]["choose_frame_for_video"]

        self.roi = None
        self.current_frame = None
        self.video = None
        self.object_data = None

    def get_video_files(self):
        """Get a list of video files in the folder."""
        if not os.path.isdir(self.folder_path):
            raise NotADirectoryError(f"folder_path is not a valid directory: {self.folder_path}")

        video_files = [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))
        ]
        return video_files

    def get_image_files(self):
        """Get a list of image files in the folder."""
        if not os.path.isdir(self.folder_path):
            raise NotADirectoryError(f"folder_path is not a valid directory: {self.folder_path}")

        image_files = [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ]
        return image_files

    def load_video(self, filepath):
        """Load a video file."""
        self.video = cv2.VideoCapture(filepath)
        if not self.video.isOpened():
            raise ValueError(f"Unable to open the video file: {filepath}")

    def choose_frame(self, frame_number=0):
        """Select a specific frame from the video."""
        if self.video is None:
            raise ValueError("No video file loaded. Cannot choose a frame.")

        # Jump to the specified frame number
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video.read()
        if ret:
            self.current_frame = frame
        else:
            raise ValueError(f"Cannot read frame {frame_number} from video.")

    def select_roi(self, image):
        """Allow the user to define a region of interest (ROI) on an image."""
        # This opens an interactive window. Press ENTER or SPACE to confirm ROI.
        roi = cv2.selectROI("Select ROI", image, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        return roi

    def define_roi_for_video(self, filepath):
        """Define ROI for a specific video."""
        self.load_video(filepath)
        frame_number = self.choose_frame_for_video
        self.choose_frame(frame_number)
        print(f"Displaying frame {frame_number} from video: {filepath}")

        self.roi = self.select_roi(self.current_frame)
        return self.roi

    def define_roi_for_image(self, filepath):
        """Define ROI for a specific image."""
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Unable to read image file: {filepath}")

        self.current_frame = image
        print(f"Displaying image: {filepath}")

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
        """
        Filter objects in the ROI for a specific (image_name).
        Note: image_name in CSV might differ from the actual file name
        if you have suffixes like '_first_frame'.
        Adjust parsing below to match your naming scheme.
        """
        if self.roi is None:
            raise ValueError("ROI not defined. Please define an ROI before filtering objects.")

        if self.object_data is None:
            raise ValueError("CSV data not loaded. Please load the CSV before filtering objects.")

        x, y, w, h = self.roi
        x_start, y_start = x, y
        x_end, y_end = x + w, y + h

        # Example filter: we expect columns ['image_name', 'x', 'y'] in your CSV
        filtered_data = self.object_data[
            (self.object_data['image_name'] == image_name) &
            (self.object_data['x'] >= x_start) & (self.object_data['x'] <= x_end) &
            (self.object_data['y'] >= y_start) & (self.object_data['y'] <= y_end)
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
        """Process all videos and images in the folder."""
        video_files = self.get_video_files()
        image_files = self.get_image_files()

        # Load CSV data once
        self.load_csv()

        all_filtered_data = pd.DataFrame()

        # --- Process images ---
        if image_files:
            print(f"Found {len(image_files)} image(s) in {self.folder_path}.")
            for image_file in image_files:
                # Define ROI for each image
                self.define_roi_for_image(image_file)
                print(f"ROI for image {image_file}: {self.roi}")

                # Remove extension and any suffix like "_first_frame" if needed
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                # If you specifically want to remove "_first_frame" (adjust if needed)
                base_name = base_name.replace("_first_frame", "")

                # Filter objects inside this ROI
                filtered_data = self.filter_objects_in_roi(base_name)
                all_filtered_data = pd.concat([all_filtered_data, filtered_data], ignore_index=True)

        # --- Process videos ---
        if video_files:
            print(f"Found {len(video_files)} video(s) in {self.folder_path}.")
            for video_file in video_files:
                image_name = os.path.basename(video_file)
                print(f"Processing video: {image_name}")
                self.define_roi_for_video(video_file)
                print(f"ROI for {image_name}: {self.roi}")

                # Remove extension from the video filename before comparing with 'image_name' in CSV
                base_name = os.path.splitext(image_name)[0]
                filtered_data = self.filter_objects_in_roi(base_name)
                all_filtered_data = pd.concat([all_filtered_data, filtered_data], ignore_index=True)

        # If neither images nor videos were found:
        if not image_files and not video_files:
            print(f"No image or video files found in {self.folder_path}.")
            return

        # Save all results to a single file
        self.save_filtered_data(all_filtered_data, filename="roi_results.csv")


# Instantiate and run the ROIDefiner class
if __name__ == "__main__":
    config_path = "configs/roi.yaml"
    roi_definer = ROIDefiner(config_path)
    roi_definer()
