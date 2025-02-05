import os
import cv2
from typing import Dict
from sahi.predict import predict as sahi_predict

class save_sahi_animation:
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = os.path.join(config['output_dir'], config['model']['task'])
        os.makedirs(self.output_dir, exist_ok=True)
        self.sahi_params = self._configure_sahi_params()
    
    def _configure_sahi_params(self) -> Dict:
        """Extracts SAHI-related parameters from the configuration."""
        return {
            "model_type": "ultralytics",
            "model_path": self.config['model']['weights'],
            "model_device": "cuda:0",
            "model_confidence_threshold": self.config['model'].get('conf_threshold', 0.2),
            "source": self.config['images_dir'],
            "slice_height": self.config['model'].get('slice_height', self.config.get('sahi', {}).get('slice_size', 640)),
            "slice_width": self.config['model'].get('slice_width', self.config.get('sahi', {}).get('slice_size', 640)),
            "overlap_height_ratio": self.config['model'].get('overlap_height_ratio', self.config.get('sahi', {}).get('overlap_ratio', 0.2)),
            "overlap_width_ratio": self.config['model'].get('overlap_width_ratio', self.config.get('sahi', {}).get('overlap_ratio', 0.2)),
        }
    
    def process_images(self):
        """Processes images using SAHI prediction."""
        sahi_predict(**self.sahi_params)
        print("Image processing complete.")
    
    def process_frame(self, frame):
        """Processes a single frame using SAHI prediction and returns the annotated frame."""
        temp_frame_path = os.path.join(self.output_dir, "temp_frame.jpg")
        cv2.imwrite(temp_frame_path, frame)
        
        self.sahi_params['source'] = temp_frame_path
        sahi_predict(**self.sahi_params)
        
        annotated_frame = cv2.imread(temp_frame_path)
        os.remove(temp_frame_path)
        return annotated_frame
    
    def process_video(self):
        """Extracts frames at vid_stride, runs SAHI prediction, and saves annotated video."""
        source_path = self.config['images_dir']
        raw_video_path = os.path.join(self.output_dir, source_path.split('/')[-1])
        output_video_path = os.path.join(self.output_dir, "annotated_video.mp4")
        
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Unable to open video: {source_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_stride = self.config['model'].get('vid_stride', 1)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        raw_writer = cv2.VideoWriter(raw_video_path, fourcc, fps, (width, height))
        
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index % vid_stride == 0:
                raw_writer.write(frame)
            
            frame_index += 1
        
        cap.release()
        raw_writer.release()
        
        self.sahi_params['source'] = raw_video_path
        sahi_predict(**self.sahi_params)
        
        os.remove(raw_video_path)
        print(f"Annotated video saved: {output_video_path} and original video deleted.")
    
    def run(self):
        """Main function to process images or videos based on input type."""
        source_path = self.config['images_dir']
        if os.path.isdir(source_path):
            self.process_images()
        elif os.path.isfile(source_path) and source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self.process_video()
        else:
            print(f"Source {source_path} is not recognized as a valid image directory or video file.")
