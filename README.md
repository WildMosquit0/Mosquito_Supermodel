# 🦟 Mosquito Supermodel

**Universal YOLO-based mosquito detection, slicing, tracking, and behavioral analysis pipeline**

This repository provides a flexible, end-to-end deep learning pipeline for detecting and analyzing mosquito behavior using YOLOv11 with support for slicing, multi-video tracking, and postprocessing.

---

## 🚀 Features

- 🔍 **Inference** with YOLOv11
- 🧩 **SAHI slicing** for small-object detection
- 🧠 **Track ID continuity** across frames/videos
- 📊 **Behavioral metrics**: visit count, duration, distance (works only with tracking mode)
- 📁 **Config-based execution** (no hardcoded paths)
- 📈 **Plotting & heatmap visualization**


## Setup Instructions


###  Pip (alternative)
```bash
python -m venv venv
source venv/bin/activate
cd path/to/Mosquito_Supermodel
pip install -r requirements.txt
```

---

## ⚙️ Configuration

All processing is driven by YAML config files in the `configs/` folder:

- `infer.yaml`: controls model weights, input path, and slicing
- `analyze.yaml`: defines behavioral analysis rules

---

## 🧠 Inference and Analysis Pipeline

### 🔹 `infer` task

The `infer` task runs YOLO-based detection (optionally with SAHI slicing) on either a **single video** or a **folder containing multiple videos**.

**Expected input structure (for batch mode):**
```
input_folder/
├── deet_rep1.mp4
├── deet_rep2.mp4
├── deet_rep3.mp4
├── control_rep1.mp4
├── control_rep2.mp4
├── control_rep3.mp4
```

Each video must follow the format:
```
treatment_repX.x
```

This naming convention helps automatically assign treatment and replicates during analysis.

**Config option for automation:**
```yaml
change_analyze_conf: true
```

If enabled, the inference process will **automatically update `configs/analyze.yaml`** with the correct output paths — so you can run analysis without editing anything manually.

---

### 🔹 `analyze` task

The `analyze` task processes the inference outputs to compute behavioral summaries.

**Included features:**

- 📊 Average number of visits per time interval
- ⏱️ Sum or average **duration** of visits
- 📏 Sum or average **distance** traveled
- 🌡️ **Heatmaps** of visit concentration
- 🔁 **X vs Y** scatter plots of mosquito positions

Results are exported as a merged `.csv` file and relevant plots.

---

## 🧠 Usage

### Run Inference
```bash
python main.py --task_name infer
```

### Run Analysis
```bash
python main.py --task_name analyze
```

Both tasks use your specified configuration in the `configs/` folder.

---

## 📁 Output Files

- `.csv` files: detection or tracking output
- `results.csv`: merged summary
- `videos/`, `frames/`, `csvs/`: organized subfolders
- Heatmaps and summary plots (if analysis is enabled)

---

## 👤 Authors

Developed by Evyatar Sar-Shalom and Ziv Kassner.  
This branch was cleaned and prepared specifically for Clément’s use.

---
