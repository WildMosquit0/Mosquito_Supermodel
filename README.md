# 🦟 Mosquito Supermodel

**Universal YOLO-based mosquito detection, slicing, tracking, and behavioral analysis pipeline**

This repository provides a flexible, end-to-end deep learning pipeline for detecting and analyzing mosquito behavior using YOLOv11, with support for slicing, multi-video tracking, and postprocessing.

---

## Features

-  **Inference** with YOLOv11
-  **SAHI slicing** for small-object detection
-  **Track ID continuity** across frames/videos
-  **Behavioral metrics** (visit count, duration, distance — available in tracking mode)
-  **Config-based execution** (no hardcoded paths)
-  **Plotting & heatmap visualization**

---

## 📦 Setup Instructions

### Pip
```bash
python -m venv venv
source venv/bin/activate
cd path/to/Mosquito_Supermodel
pip install -r requirements.txt
```

---

## ⚙️ Configuration

All operations are driven by YAML files in the `configs/` folder:

- `infer.yaml`: defines model weights, paths, task type
- `analyze.yaml`: defines analysis logic

---

## Inference and Analysis Pipeline

### 🔹 `infer` task

The `infer` task performs object detection or tracking on a single video or a folder of videos.

#### 🔧 `infer.yaml` structure:
```yaml
images_dir: path/to/video/or/folder
model:
  weights: path/to/model.pt
  conf_threshold: confidence threshold for predictions
  iou_threshold: IoU threshold for NMS
  task: track / predict / slice
  vid_stride: 5  # Predict every Nth frame. Lower = more accurate tracking

output_dir: path/to/save/project  # Created automatically if not exists

sahi:
  slice_size: 640          # Slice each frame into 640×640 patches (recommended for this model)
  overlap_ratio: 0.2       # 20% overlap between adjacent slices for better detection coverage
  track: true              # Enable tracking across sliced frames

save_animations: true      # Save predicted video
change_analyze_conf: true  # Automatically update configs/analyze.yaml
```

#### 📂 Expected input format for batch mode:
```
input_folder/
├── deet_rep1.mp4
├── deet_rep2.mp4
├── control_rep1.mp4
...
```

Each video should be named as:
```text
<treatment>_repX.mp4
```

When `change_analyze_conf: true`, the analyzer config is automatically updated based on inference results.

---

### 🔹 `analyze` task

The `analyze` task processes output from inference and computes behavioral metrics.

#### 🔧 `analyze.yaml` structure:
```yaml
input_csv: path/to/inference/results.csv  # Auto-filled if infer used with change_analyze_conf: true
output_dir: path/to/output/folder

settings:
  interval_unit: minute  # or 'seconds'
  filter_time_intervals: 15  # Limit duration of analysis
  fps: 25  # Original FPS ÷ vid_stride

  stat: sum  # How to summarize: sum, mean, or median
  time_intervals: 1  # Time binning (e.g. every 1 min)
  treatment_or_image_name: treatment  # Use treatment or replicates in plots

heatmap:
  grid_size: 30  # Higher = finer resolution (smaller grid cells)
  image_path: path/to/project/frames
  min_count: 1  # Minimum visits to display
  true_axis: true  # Plot in real pixel space

plotxy:
  id_OR_class: class  # 'id' = unique trajectories, 'class' = object type
  treatment_or_image_name: image_name
  true_axis: true

task:
  distance: true
  duration: true
  heatmap: true
  plotxy: true
  visits: true
```

---

## 📈 Analysis Outputs

- **Visits** per time interval
- **Duration** of object presence
- **Distance** traveled
- **Heatmaps** showing visit density
- **X vs Y scatter plots** of object positions

All results are saved as `.csv` summaries and visual plots in the configured output directory.

---

## Usage

### Run Inference
```bash
python main.py --task_name infer
```

### Run Analysis
```bash
python main.py --task_name analyze
```

---

## 📁 Output Structure

- `results.csv`: merged behavior metrics
- `videos/`, `frames/`, `csvs/`: organized intermediate outputs
- `.png` plots: for visits, heatmaps, trajectories




