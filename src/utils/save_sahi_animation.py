import os
import cv2
from typing import Dict
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

def save_sahi_animation(config: Dict) -> None:
    """
    Runs SAHI sliced prediction on the source—if it's an image directory, it annotates each image;
    if the source is a video, it processes every nth frame (according to vid_stride), annotates those
    frames, and writes them into a video file.
    
    All parameters (model, confidence, slicing parameters, and vid_stride) are read from the configuration.
    """
    # Load configuration parameters.
    model_path = config['model']['weights']
    conf_threshold = config['model'].get('conf_threshold', 0.2)
    device = "cuda:0"  # Optionally check torch.cuda.is_available()

    # Get slicing parameters. Try first the model section, then fallback to sahi section.
    slice_height = config['model'].get('slice_height', config.get('sahi', {}).get('slice_size', 640))
    slice_width  = config['model'].get('slice_width',  config.get('sahi', {}).get('slice_size', 640))
    overlap_height_ratio = config['model'].get('overlap_height_ratio', config.get('sahi', {}).get('overlap_ratio', 0.2))
    overlap_width_ratio  = config['model'].get('overlap_width_ratio',  config.get('sahi', {}).get('overlap_ratio', 0.2))
    
    # Video stride parameter.
    vid_stride = config['model'].get('vid_stride', 1)
    
    source_path = config['images_dir']
    output_dir = os.path.join(config['output_dir'], config['model']['task'])
    os.makedirs(output_dir, exist_ok=True)

    # Load the SAHI detection model.
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device=device
    )

    # If the source is a directory, assume it contains images.
    if os.path.isdir(source_path):
        file_list = [os.path.join(source_path, f) for f in os.listdir(source_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for file_path in file_list:
            image = cv2.imread(file_path)
            if image is None:
                print(f"Warning: Unable to load image {file_path}")
                continue

            # Run SAHI sliced prediction.
            result = get_sliced_prediction(
                image=image,
                detection_model=detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio
            )

            # Annotate the image (draw bounding boxes).
            annotated_image = image.copy()
            for obj_pred in result.object_prediction_list:
                x, y, w, h = obj_pred.bbox.to_xywh()
                cv2.rectangle(annotated_image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

            image_name = os.path.basename(file_path)
            # Ensure the filename has an extension.
            if not os.path.splitext(image_name)[1]:
                image_name += ".jpg"
            save_path = os.path.join(output_dir, image_name)
            cv2.imwrite(save_path, annotated_image)
            print(f"Saved annotated image: {save_path}")

    # Else, if the source is a video file.
    elif os.path.isfile(source_path) and source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Unable to open video: {source_path}")
            return

        # Get video properties.
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret:
            print("No frames found in video.")
            cap.release()
            return
        height, width = frame.shape[:2]
        # Prepare a VideoWriter to combine annotated frames.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(output_dir, "annotated_video.mp4")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Reset the video to the beginning.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process only frames based on the vid_stride.
            if frame_index % vid_stride != 0:
                frame_index += 1
                continue

            # Run SAHI sliced prediction.
            result = get_sliced_prediction(
                image=frame,
                detection_model=detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio
            )
            annotated_frame = frame.copy()
            for obj_pred in result.object_prediction_list:
                x, y, w, h = obj_pred.bbox.to_xywh()
                cv2.rectangle(annotated_frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            writer.write(annotated_frame)
            print(f"Processed frame {frame_index}")
            frame_index += 1

        cap.release()
        writer.release()
        print(f"Saved annotated video: {output_video_path}")

    else:
        print(f"Source {source_path} is not recognized as a valid image directory or video file.")
