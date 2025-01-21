
# Mosquito Supermodel

The **Mosquito Supermodel** project aims to create a general detector for mosquitoes (any species) using the YOLO framework. This standalone open-source project provides accessible tools, including detector weights and training data, for anyone interested in detecting and exploring mosquito behaviors.

---

## Table of Contents
1. [Overview](#overview)
2. [Setup and Installation](#setup-and-installation)
3. [Usage](#usage)
4. [Features](#features)
5. [File Structure](#file-structure)
6. [Contact](#contact)
7. [Future Enhancements](#future-enhancements)

---

## Overview

The **Mosquito Supermodel** project is designed for researchers, developers, and the general public, offering tools to detect mosquitoes and analyze their behavior. It is entirely open source, allowing anyone to:
- Access the detector weights.
- Explore and use the training data.
- Perform mosquito detection and behavioral analysis.

---

## Setup and Installation

### Prerequisites
- Python (version TBD)
- Libraries specified in `requirements.txt`.
- OpenCV library (install separately).

### Installation
1. Clone this repository:
   ```bash
   git clone -b feature/analyzer https://github.com/WildMosquit0/Mosquito_Supermodel.git
   cd Mosquito_Supermodel
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install opencv-python
   ```

---

## Usage

### Configuration
- The repository includes a JSON configuration file that controls all arguments, including:
  - Prediction types (`predict` or `track`).
  - Analysis parameters.
- Default values are provided but can be adjusted for specific use cases, such as varying video frame rates.

### Running the Project
- To predict or track:
  ```bash
  python main.py
  ```
- Input formats:
  - Video folder for prediction.
  - CSV file for analysis.
- Outputs:
  - CSV files.
  - Plots.

---

## Features

1. **Prediction and Tracking**
   - Framework functions to predict and track mosquito movements using YOLOv11.

2. **Analysis**
   - Tools for analyzing YOLOv8 output, providing detailed insights through CSV files and plots.

---

## File Structure

- **Main Files and Directories**:
  - `vids/` or `images/`: Input video or image files.
  - `main.py`: The main script for running predictions and analyses.
  - Configuration files: JSON files controlling various project parameters.

- **Modifications**:
  - Users can modify any file if needed; there are no strict restrictions.
  - Clear separation of functionality may be updated over time.

---

## Contact

For questions or issues, feel free to reach out via email:

**Evyatar Sar-Shalom**  
📧 [evyatar.sar-shalom@mail.huji.ac.il](mailto:evyatar.sar-shalom@mail.huji.ac.il)

---

## Future Enhancements

- **Planned Features**:
  - Uploading links to datasets and weights.
  - Introducing a GUI interface for easier use.

- **Project Updates**:
  - Updates will occur occasionally.
  - Users are encouraged to create their own branches for contributions or customizations.

---
