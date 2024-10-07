import csv
import os
from typing import List
from ultralytics.engine.results import Results


class ResultsParser:
    def __init__(self, results: List[Results], output_dir: str = './output', task: str = 'detection'  ) -> None:
        self.results = results
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def parse_and_save(self):
        csv_file_path = os.path.join(self.output_dir, 'results.csv')

        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['image_idx', 'image_name', 'box_idx', 'x1', 'y1', 'x2', 'y2', 'confidence', 'label', 'track_id'])

            for img_idx, result in enumerate(self.results):
                image_name = os.path.basename(result.path)  
                boxes = result.boxes.xyxy.cpu().numpy()  
                scores = result.boxes.conf.cpu().numpy() 
                labels = result.boxes.cls.cpu().numpy()  
                track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None  

                for box_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    track_id = track_ids[box_idx] if track_ids is not None else None
                    writer.writerow([img_idx, image_name, box_idx, *box, score, label, track_id])

        print(f"Results saved to {csv_file_path}")


if __name__ == "__main__":
    parser = ResultsParser(results, output_dir='./output', csv_filename='results.csv')
    parser.parse_and_save()
