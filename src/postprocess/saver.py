import csv
import os
import yaml
from typing import List
from ultralytics.engine.results import Results
from src.utils.sahi_usage import sahi_usage  # Adjust import based on your project structure

class ResultsParser:
    def __init__(self, results: List[Results], config: dict):
        self.results = results
        self.output_dir = os.path.join(config['output_dir'], config['model']['task'])
        self.csv_filename = config.get('csv_filename', 'results.csv')
        self.images_dir = config.get('images_dir')
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def parse_and_save(self):
        """
        Save ultralytics results to CSV. Assumes each result object has a .path attribute.
        CSV columns: image_idx, box_idx, x, y, w, h, confidence, label, track_id, image_name, img_h, img_w.
        """
        csv_file_path = os.path.join(self.output_dir, self.csv_filename)
        image_idx_tracker = {}

        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                'image_idx', 'box_idx', 'x', 'y', 'w', 'h', 
                'confidence', 'label', 'track_id', 'image_name', 
                'img_h', 'img_w'
            ])

            for result in self.results:
                # Ultraytics results: result.path should exist.
                image_name_with_idx = os.path.basename(result.path)
                image_name = image_name_with_idx.split(".")[0]

                if image_name not in image_idx_tracker:
                    image_idx_tracker[image_name] = 0

                img_idx = image_idx_tracker[image_name]
                boxes = result.boxes.xywh.cpu().numpy()
                original_height, original_width = result.orig_shape
                scores = result.boxes.conf.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None

                for box_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    track_id = track_ids[box_idx] if track_ids is not None else None
                    writer.writerow([
                        img_idx, box_idx, *box, score, label, 
                        track_id, image_name, original_height, original_width
                    ])

                image_idx_tracker[image_name] += 1

        print(f"Results saved to {csv_file_path}")

    def parse_and_save_slice(self, predictions: List[tuple]):
        """
        Save SAHI-based predictions to CSV. Assumes predictions are provided as a list of tuples,
        each in the form:
          (box_idx, x, y, w, h, confidence, source_identifier, frame_index, img_h, img_w)
        Since SAHI predictions do not include label or track_id, these columns are set to default values.
        CSV columns: image_idx, box_idx, x, y, w, h, confidence, label, track_id, image_name, img_h, img_w.
        """
        csv_file_path = os.path.join(self.output_dir, self.csv_filename)
        image_idx_tracker = {}

        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                'image_idx', 'box_idx', 'x', 'y', 'w', 'h', 
                'confidence', 'label', 'track_id', 'image_name', 
                'img_h', 'img_w'
            ])
            for pred in predictions:
                (box_idx, x, y, w, h, confidence,
                 source_identifier, frame_index, img_h, img_w) = pred

                if source_identifier not in image_idx_tracker:
                    image_idx_tracker[source_identifier] = 0
                image_idx = image_idx_tracker[source_identifier]
                label = ""     # Default label
                track_id = None  # Default track id

                writer.writerow([
                    image_idx, box_idx, x, y, w, h, confidence, label, track_id,
                    source_identifier, img_h, img_w
                ])
                image_idx_tracker[source_identifier] += 1

        print(f"Results saved to {csv_file_path}")


# --- Example Usage ---
if __name__ == "__main__":
    with open("configs/infer.yaml", "r") as f:
        config = yaml.safe_load(f)
    # If you have ultralytics Results from a previous process:
    # results = ... 
    # rp = ResultsParser(results, config)
    # rp.parse_and_save()
    
    # For demonstration using SAHI-based predictions:
    from src.utils.sahi_usage import sahi_usage
    su = sahi_usage(config)
    # Run SAHI inference once to obtain predictions.
    predictions = su.run_command(config["images_dir"])
    rp = ResultsParser([], config)
    rp.parse_and_save_slice(predictions)
