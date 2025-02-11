import os
import mimetypes
import cv2

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


import cv2
import os
from typing import Union, List

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