# Gait Analysis System

A computer vision-based gait analysis system that extracts human joint angles and movement patterns from video input using pose estimation techniques.

---

## Overview

This project analyzes human walking patterns (gait) from videos by detecting body landmarks and calculating joint angles. It supports both **side-view** and **front-view** videos and generates structured data for further analysis or model training.

---

## Features

- 🎥 Supports **video input (MP4)**
- 🧍 Pose estimation using **MediaPipe**
- 📐 Calculates joint angles (hip, knee, ankle, etc.)
- 🔄 Works with **side-view and front-view videos**
- 📊 Exports data to **CSV for further analysis**
- 🖥️ Visual overlay of angles on output video
- ⚙️ Modular pipeline (easy to extend)

---

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Matplotlib

---

## Project Structure
```
📦 gait-analysis/
├─ data/
│  ├─ raw_videos
│  └─ processed
├─ src/
│  ├─ main.py
│  ├─ pose_estimation.py
│  ├─ angle_calculation.py
│  ├─ gait_features.py
│  └─ csv_writer.py
├─ results/
│  ├─ angle_plot.png
│  ├─ front_view_plot.png
│  └─ gait_data.csv
├─ .gitignore
├─ .gitattributes
├─ requirements.txt
├─ README.md
└─ Figure_1.png
```


---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/gait-analysis.git
cd gait-analysis
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
1. Place your input video in:
```bash
data/raw_videos/
```

2. Update video path in main.py (if needed)
3. Run the pipeline:
```bash
python src/main.py
```

## Output
The system generates:

#### Processed Video
- Annotated with:
    - Body landmarks
    - Joint angles
#### CSV File
Contains frame-wise data:
```bash
frame, hip_angle, knee_angle, ankle_angle, ...
```

## Applications
- Medical gait analysis
- Sports performance tracking
- Training ML models on motion data
- Rehabilitation monitoring

## Limitations
- Accuracy depends on:
    - Video quality
    - Camera angle
- Best results with:
    - Clear background
    - Proper lighting
- Currently supports single-person tracking

## Future Improvements
- Multi-person tracking
- Real-time analysis (webcam)
- Integration with ML models for anomaly detection
- GUI/dashboard for visualization
- Support for 3D pose estimation