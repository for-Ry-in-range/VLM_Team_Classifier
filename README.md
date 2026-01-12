# VLM Team Classifier

A computer vision system that uses Vision Language Models (VLMs) and unsupervised clustering to classify people in basketball footage as Team A, Team B, and referees. 

## Overview

The system uses SigLIP and K-Means clustering to classify players on a basketball court into:
- Team A (displayed in blue)
- Team B (displayed in red)  
- Referees (displayed in yellow)

Summary of how the system works:
1. Detects people with YOLO object detection
2. Filters out audience members and coaches on the sideline
3. Extracts embeddings with SigLIP
4. Clusters embeddings to identify teams and referees
5. Classifies players and referees

## Prerequisites

- Python 3.10 or higher
- A video of basketball gameplay
- GPU

## Installation

1. Clone the repository

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r ../requirements.txt
   ```


4. Download YOLO weights
   (`yolov8n.pt` will automatically be downloaded on the first execution.)

## Usage

### Basic Usage

1. Put your video file in the `vlm_team_classifier` directory

2. Edit the video path in `run_video.py`
   ```python
   if __name__ == "__main__":
       main("warriors_bucks.mp4")  # Change to your video's filename
   ```

3. Run this script:
   ```bash
   python run_video.py
   ```

### What Should Happen

1. Initialization:
   - The script loads YOLO
   - Loads the SigLIP model
   - Searches for a frame with at least 9 people on the court

2. Training:
   - Once a frame is found, embeddings are extracted from all detected people
   - Groups people into 3 clusters (Team A, Team B, Referees) with K-Means clustering
   - The smallest cluster is assigned as the referees

3. Classification:
   - Processes each frame
   - Detects people with YOLO
   - Classifies each person as Team A, Team B, or Referee
   - Adds colored outlines over video to display classifications

### Controls

- Press Q to quit

## Requirements for Video

- The video must show basketball gameplay
- At least one frame must have at least 9 people on the court

## Output

A real-time video window showing:
- Color coded bounding boxes around each detected person on the court
- Classification labels on top of each box

## Alternative: Baseline Classifier

For running the simpler baseline classifier:
```bash
python run_baseline.py
```

This only uses average jersey colors.
