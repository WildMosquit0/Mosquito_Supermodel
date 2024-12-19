import csv
import os
from typing import List
from ultralytics.engine.results import Results


class ResultsParser:
    def __init__(self, results: List[Results], config: dict) -> None:
        self.results = results
        self.output_dir = config['output']['output_dir']
        self.csv_filename = config['output'].get('csv_filename', 'results.csv')  

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def parse_and_save(self):
        csv_file_path = os.path.join(self.output_dir, self.csv_filename)

        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['image_idx', 'box_idx', 'x', 'y', 'w', 'h', 'confidence', 'label', 'track_id','image_name','img_h','img_w'])

            for img_idx, result in enumerate(self.results):
<<<<<<< HEAD
                image_name_with_inx = os.path.basename(result.path)
                image_name = image_name_with_inx.split(".")[0]
                boxes = result.boxes.xywh.cpu().numpy() 
                original_height,original_width = result.orig_shape
                scores = result.boxes.conf.cpu().numpy() 
                labels = result.boxes.cls.cpu().numpy()  
                track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None 
=======
                image_name = os.path.basename(result.path)
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                scores = result.boxes.conf.cpu().numpy()  # Confidence scores
                labels = result.boxes.cls.cpu().numpy()  # Labels (class indices)
                track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None  # Track IDs for tracking tasks
>>>>>>> 0738f393f5a7ca9b630af2796d119c284a60f75a

                for box_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    track_id = track_ids[box_idx] if track_ids is not None else None
                    writer.writerow([img_idx, box_idx, *box, score, label, track_id, image_name, original_height,original_width])

        print(f"Results saved to {csv_file_path}")
<<<<<<< HEAD

=======
>>>>>>> 0738f393f5a7ca9b630af2796d119c284a60f75a
