# 🏀 ShotIQ — AI Basketball Shot Analysis System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=white)](https://ultralytics.com)

**ShotIQ** is a high-performance, offline video-analysis system designed to transform raw basketball footage into professional-grade sports analytics. By leveraging **YOLOv8** computer vision and advanced physics modeling, ShotIQ provides cinematic overlays that track trajectory, predict outcomes, and visualize the science of every shot.

---

## ✨ Features

*   **🔍 Precision Ball Detection:** Uses a custom-tuned YOLOv8 model for stable basketball detection even in challenging or dim lighting conditions.
*   **📊 Cinematic Analytics Overlays:** Professional-grade telemetry panels showing real-time speed, target position, and AI confidence.
*   **📈 Smooth Trajectory Tracking:** Implements a filtered Kalman tracking system for fluid, jitter-free ball paths and glowing motion trails.
*   **🎯 Predictive Intelligence:** Real-time trajectory projection that determines if a shot is "On Target" before it reaches the hoop.
*   **🎬 Automated Video Export:** Processes standard MP4 clips frame-by-frame and renders a polished analytics video.
*   **🛠️ Adaptive Physics:** Auto-scaling gravity and detection parameters based on video resolution (720p, 1080p, 4K).

---

## 🚀 Tech Stack

*   **Logic:** [Python 3.8+](https://www.python.org/)
*   **Computer Vision:** [OpenCV](https://opencv.org/)
*   **AI/Deep Learning:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
*   **Data Processing:** [NumPy](https://numpy.org/)
*   **Tracking:** Kalman Filter (Linear Dynamics)

---

## 🏗️ Architecture & Workflow

ShotIQ follows a modular pipeline designed for maximum visual quality and predictive accuracy:

1.  **Input:** Load basketball footage (MP4/MOV).
2.  **Detection Layer:** YOLOv8 detects the sports ball class with a low-confidence threshold for dim-light robustness.
3.  **Tracking Layer:** A Kalman Filter predicts the ball's position during motion blur or occlusions, generating a stabilized history.
4.  **Heuristic Engine:** Analyzes pixel-velocity to detect "Ball in Hand," "Shot Release," and release velocity.
5.  **Prediction Engine:** Uses gravity-based kinematics to project the 2D parabolic arc and check for hoop intersection.
6.  **Rendering:** Applies alpha-blended glowing trails, telemetry panels, and state-driven labels (e.g., *predicting...*, *will go in*).
7.  **Output:** Saves the finalized analytics footage.

---

## ⚙️ Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/shotiq.git
    cd shotiq
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download YOLOv8 Weights:**
    The system uses `yolov8n.pt` (nano). It will be downloaded automatically on the first run, or you can place it in the project root.

---

## 🛠️ Usage

To analyze a video and export the cinematic output, run:

```bash
python main.py --input path/to/your_shot.mp4 --output shotiq_analyzed.mp4
```

(you may use the sample video shot1.mp4 for demonstration. It is in the same directory as this file.)

### ⌨️ Key Parameters
*   `--input`: (Required) Path to the input basketball video file.
*   `--output`: (Optional) Name of the analyzed video file (default: `shotiq_analyzed.mp4`).

---

## 📁 Folder Structure

```text
ShotIQ/
├── detector.py      # YOLOv8 implementation & detection stability
├── tracker.py       # Kalman Filter logic & position history
├── trajectory.py    # Gravity-based physics & intersection prediction
├── utils.py         # Cinematic drawing utilities & UI panels
├── config.py        # Centralized tuning (Confidence, Gravity, Colors)
├── main.py          # Application entry point & processing loop
└── requirements.txt # Project dependencies
```

---

## 🔮 Future Improvements

*   **🏀 Dynamic Hoop Detection:** Fully automated hoop tracking for moving cameras/drones.
*   **📈 Advanced Biometrics:** Integration of pose estimation for shooting form analysis (elbow angle, jump height).
*   **📱 Mobile Integration:** Optimization for on-device analysis.
*   **📊 Multi-Shot Stats:** Session-wide analytics reporting (Shot map, consistency scores).


