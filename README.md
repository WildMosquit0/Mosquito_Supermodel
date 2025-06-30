# 🦟 Mosquito Supermodel

**Universal YOLO-based mosquito detection, slicing, tracking, and behavioral analysis pipeline**

This repository provides a flexible, end-to-end deep learning pipeline for detecting and analyzing mosquito behavior using YOLO models (e.g., YOLOv8) with support for slicing, multi-video tracking, and postprocessing. Originally built for mosquito research, the tools are adaptable to similar tasks in entomology and small-object behavior tracking.

---

## 🚀 Features

- 🔍 **Inference** with YOLOv8 or YOLOv11
- 🧩 **SAHI slicing** for small-object detection
- 🧠 **Track ID continuity** across frames/videos
- 📊 **Behavioral metrics**: visit count, duration, distance
- 📁 **Config-based execution** (no hardcoded paths)
- 📈 **Plotting & heatmap visualization**
- ⚙️ Modular design: `inference`, `analyze`, `train`, `track`, and `utils`

---

## 🗂️ Repository Structure

```
Mosquito_Supermodel/
├── configs/              # YAML config files for inference, analysis, and ROI
├── src/                  # Source code modules
│   ├── inference/        # YOLO + SAHI inference logic
│   ├── analyze/          # Visit/distance analysis
│   ├── tracking/         # Track ID handling
│   ├── utils/            # Common helpers
├── main.py               # Entry point for running inference or analysis
├── requirements.txt      # Pip-based dependencies
├── environment.yml       # Conda environment
└── README.md             # You're here!
```

---

## 🧪 Setup Instructions

### 🔧 Conda (recommended)
```bash
conda env create -f environment.yml
conda activate super_model
```

### 📦 Pip (alternative)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Configuration

All processing is driven by YAML config files in the `configs/` folder:

- `infer.yaml`: controls model weights, input path, and slicing
- `analyze.yaml`: defines behavioral analysis rules
- `roi.yaml`: region-of-interest parameters (if needed)

---

## 🧠 Inference and Analysis Pipeline

### 🔹 `infer` task

The `infer` task runs YOLO-based detection (optionally with SAHI slicing) on either a **single video** or a **folder containing multiple videos**.

**Expected input structure (for batch mode):**
```
input_folder/
├── deet_rep1.avi
├── control_rep2.avi
├── eugenol_rep3.avi
```

Each video must follow the format:
```
<treatment>_repX.avi
```

This naming convention helps automatically assign treatment labels during analysis.

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

Developed by Evyatar Sar-Shalom and collaborators.  
This branch was cleaned and prepared specifically for Clément’s use.

---

## 📜 License

MIT License (or specify your own).
