import os
import mimetypes
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


