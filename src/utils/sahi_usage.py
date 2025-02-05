import os
import cv2
import csv
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

class sahi_usage:
    def __init__(self, config: dict):
        # Model settings from YAML "model" section.
        self.model = config['model']['weights']
        self.task = config['model']['task']
        self.conf_threshold = config['model'].get('conf_threshold', 0.3)
        self.iou_threshold = float(config['model'].get('iou_threshold', 0.25))
        self.vid_strides = config['model'].get('vid_stride', 1)
        
        # SAHI-specific settings from "sahi" section.
        self.slice_size = config.get('sahi', {}).get('slice_size', 640)
        self.overlap_ratio = config.get('sahi', {}).get('overlap_ratio', 0.2)
        
        # Other settings.
        self.images_dir = config.get('images_dir')
        self.output_dir = os.path.join(config['output_dir'], self.task)
        self.csv_filename = config.get('csv_filename', 'results.csv')
        self.save_animations = config.get('save_animations', False)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # --- Model Loading ---
    def load_model(self, device="cuda:0"):
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=self.model,
            confidence_threshold=self.conf_threshold,
            device=device
        )
        return detection_model

    # --- File Utilities ---
    def get_file_list(self, source):
        if os.path.isdir(source):
            filenames = os.listdir(source)
            file_list = [os.path.join(source, f) for f in filenames]
            base_names = [os.path.splitext(f)[0] for f in filenames]
        elif os.path.isfile(source):
            file_list = [source]
            base_names = [os.path.splitext(os.path.basename(source))[0]]
        else:
            raise ValueError(f"Source {source} is neither a valid file nor a directory.")
        return file_list, base_names

    # --- Prediction Functions ---
    def sahi_predict(self, image_input, detection_model,
                     slice_height=None, slice_width=None,
                     overlap_height_ratio=None, overlap_width_ratio=None):
        if slice_height is None:
            slice_height = self.slice_size
        if slice_width is None:
            slice_width = self.slice_size
        if overlap_height_ratio is None:
            overlap_height_ratio = self.overlap_ratio
        if overlap_width_ratio is None:
            overlap_width_ratio = self.overlap_ratio

        result = get_sliced_prediction(
            image=image_input,
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio
        )
        return result.object_prediction_list

    def extract_predictions(self, image, object_predictions, source_identifier, frame_index=None):
        img_height, img_width = image.shape[:2]
        predictions = []
        for idx, obj_pred in enumerate(object_predictions):
            x, y, w, h = obj_pred.bbox.to_xywh()
            confidence = obj_pred.score.value
            source_identifier = source_identifier.split('.')[0]
            predictions.append((idx, x, y, w, h, confidence, source_identifier, frame_index, img_height, img_width))
        return predictions

    # --- Video Processing ---
    def process_video(self, video_path, detection_model):
        all_predictions = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            return all_predictions

        idx = frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % self.vid_strides != 0:
                frame_index += 1
                continue
            object_predictions = self.sahi_predict(frame, detection_model)
            source_id = os.path.basename(video_path)
            preds = self.extract_predictions(frame, object_predictions, source_id, frame_index=idx)
            all_predictions.extend(preds)
            frame_index += 1
            idx += 1
        cap.release()
        return all_predictions

    # --- NMS (Non-Maximum Suppression) ---
    def compute_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
        box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2

        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        iou = inter_area / float(area1 + area2 - inter_area)
        return iou

    def nms_for_group(self, predictions, iou_threshold):
        sorted_preds = sorted(predictions, key=lambda x: x[5], reverse=True)
        keep = []
        while sorted_preds:
            current = sorted_preds.pop(0)
            keep.append(current)
            filtered_preds = []
            for pred in sorted_preds:
                iou = self.compute_iou((current[1], current[2], current[3], current[4]),
                                       (pred[1], pred[2], pred[3], pred[4]))
                if iou < iou_threshold:
                    filtered_preds.append(pred)
            sorted_preds = filtered_preds
        return keep

    def apply_nms(self, predictions, iou_threshold=None):
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        grouped = {}
        for pred in predictions:
            key = (pred[6], pred[7])  # Group by (source_identifier, frame_index)
            grouped.setdefault(key, []).append(pred)
        final_preds = []
        for key, preds in grouped.items():
            nms_preds = self.nms_for_group(preds, iou_threshold)
            final_preds.extend(nms_preds)
        return final_preds

    # --- CSV Saving ---
    def save_predictions_to_csv(self, predictions, csv_file_path):
        headers = ['box_idx', 'x', 'y', 'w', 'h', 'confidence', 'image_idx', 'frame_index', 'img_h', 'img_w']
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            for pred in predictions:
                writer.writerow(pred)
        print(f"Predictions saved to {csv_file_path}")

    # --- Main Processing ---
    def run_command(self, source):
        detection_model = self.load_model()
        file_list, _ = self.get_file_list(source)
        all_predictions = []
        frame_index = 0
        for file_path in file_list:
            file_lower = file_path.lower()
            is_video = file_lower.endswith(('.mp4', '.avi', '.mov', '.mkv'))
            if is_video:
                print(f"Processing video: {file_path}")
                video_preds = self.process_video(file_path, detection_model)
                all_predictions.extend(video_preds)
            else:
                print(f"Processing image: {file_path}")
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Warning: Unable to read image {file_path}")
                    continue
                object_predictions = self.sahi_predict(image, detection_model)
                preds = self.extract_predictions(image, object_predictions, os.path.basename(file_path), frame_index=frame_index)
                all_predictions.extend(preds)
                frame_index += 1
        all_predictions.sort(key=lambda x: x[6])
        all_predictions = self.apply_nms(all_predictions)
        return all_predictions

# --- Example Usage ---
if __name__ == "__main__":
    import yaml
    with open("configs/infer.yaml", "r") as f:
        config = yaml.safe_load(f)
    su = sahi_usage(config)
    source = config["images_dir"]
    predictions = su.run_command(source)
    csv_output = os.path.join(os.path.dirname(source), su.csv_filename)
    su.save_predictions_to_csv(predictions, csv_output)
