# Tracking using OpenCV

This is a Python script for tracking objects in a video file using OpenCV. The script applies background subtraction to the video to isolate moving objects, then finds contours of those objects and selects only the contours that correspond to objects. Finally, it draws rectangles around the fish and lines connecting their centers to track their movement.

## Requirements

- Python 3.x
- OpenCV

## Installation

1. Install Python 3.x: https://www.python.org/downloads/
2. Install OpenCV: `pip install opencv-python`
