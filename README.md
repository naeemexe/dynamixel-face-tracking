# dynamixel-face-tracking

This project implements a real-time face tracking system using MediaPipe, OpenCV, and Dynamixel servos. A camera detects a face, calculates its offset from the center of the frame, and moves two Dynamixel motors to keep the face centered through synchronized position control.

Features:

Real-time face detection using MediaPipe

Smooth pan-tilt tracking with Dynamixel Protocol 2.0

High-speed communication at 4 Mbps

PID-tuned servo response

Safe angle limits and synchronized read/write operations

Tech Stack:
Python
OpenCV
MediaPipe
Dynamixel SDK

Overview:
The main script (camera.py) handles the full tracking loop: capture frames, detect face, compute offset, convert pixel displacement to servo angle, and send updated goal positions. The repository also includes modules for YOLO/ONNX detection, CUDA-accelerated processing, and Kalman filtering for more advanced tracking options.
