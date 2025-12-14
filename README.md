# Deep Learning Based Image Analysis System for Accurate Detection of Acupressure Points

![Acupressure Detection System](https://via.placeholder.com/1000x300?text=Deep+Learning+Acupressure+Point+Detector)

> [cite_start]**Note:** This project utilizes Computer Vision and Deep Learning to bridge traditional healing practices with modern technology, providing real-time guidance for self-acupressure[cite: 39].

## 📋 Table of Contents
- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [Supported Acupressure Points](#-supported-acupressure-points)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Model Training](#-model-training)
- [Results](#-results)
- [Future Enhancements](#-future-enhancements)
- [Contributors](#-contributors)

---

## 📖 About the Project

[cite_start]Acupressure is a widely practiced non-invasive therapy, but accurately locating specific pressure points is a challenge for beginners due to anatomical variations[cite: 30]. [cite_start]Errors in identifying these points can lead to reduced therapeutic benefits[cite: 31].

This project proposes a **Deep Learning–Based Image Analysis System** designed to automate point localization. [cite_start]It integrates **OpenCV**, **MediaPipe** hand-landmark detection, and a **Convolutional Neural Network (CNN)** based regression model to analyze real-time hand images captured through a webcam[cite: 32, 33]. [cite_start]The system predicts precise coordinates for acupressure points and overlays them on the live feed, enabling users to visualize locations with high accuracy[cite: 35].

---

## ✨ Key Features

* [cite_start]**Real-Time Detection:** Processes live video feed at approximately 20–25 FPS[cite: 38].
* [cite_start]**Hand Landmark Tracking:** Utilizes MediaPipe to extract 21 key hand landmarks for structural analysis[cite: 236, 270].
* [cite_start]**Deep Learning Prediction:** Uses a lightweight CNN trained on a custom/synthetic dataset to predict (x, y) coordinates of therapeutic points[cite: 273].
* [cite_start]**Boundary Constraints:** Algorithms ensure predicted points remain constrained within the hand boundary to account for movement[cite: 734].
* [cite_start]**Interactive Interface:** A user-friendly GUI (Tkinter/Flask) that provides visual overlays and on-screen therapeutic information[cite: 38, 338].
* [cite_start]**Robustness:** Handles variations in hand size, orientation, lighting, and camera distance[cite: 37].

---

## 📍 Supported Acupressure Points

[cite_start]The system is trained to detect and visualize the following points [cite: 350-368]:

| Point Code | Name | Indicated Condition | Color Code |
|:---:|:---:|:---:|:---:|
| **LI-4** | Hegu | Cold Relief & Headaches | 🟠 Orange |
| **PC-8** | Laogong | Stress Relief | 🔵 Light Blue |
| **HT-7** | Shen Men | Anxiety Relief | 🟢 Green |

---

## 🏗 System Architecture

[cite_start]The system follows a modular pipeline design [cite: 223-228]:

1.  **Image Acquisition:** Captures real-time frames via webcam.
2.  **Preprocessing:** Resizes images, normalizes pixel values, and applies contrast enhancement using OpenCV.
3.  **Landmark Detection:** MediaPipe extracts 21 hand landmarks to form a feature vector.
4.  **CNN Prediction:** The normalized landmarks are fed into the trained CNN model to predict acupressure point coordinates.
5.  **Visualization:** The backend maps predictions back to pixel coordinates, applies boundary constraints, and overlays the data on the UI.

---

## 🛠 Technology Stack

* [cite_start]**Language:** Python [cite: 163]
* [cite_start]**Deep Learning:** TensorFlow / Keras [cite: 336, 337]
* [cite_start]**Computer Vision:** OpenCV, MediaPipe [cite: 167, 169]
* [cite_start]**GUI Framework:** Tkinter (Desktop) or Flask (Web) [cite: 173, 338]
* [cite_start]**Data Handling:** NumPy, Pandas [cite: 177]

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python installed along with the following libraries:
* opencv-python
* mediapipe
* tensorflow
* numpy
* pillow

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/acupressure-detection.git](https://github.com/yourusername/acupressure-detection.git)
    cd acupressure-detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install opencv-python mediapipe tensorflow numpy pillow
    ```

### Usage

1.  **Train the Model (First Run):**
    Before running the main application, you must generate the model file (`acupressure_model.keras`). [cite_start]The training script generates a synthetic dataset of 50,000 samples based on canonical hand skeletons[cite: 928, 1007].
    ```bash
    python train_model.py
    ```

2.  **Launch the Detector:**
    [cite_start]Once the model is trained, run the main application to start the webcam feed and detection[cite: 344, 801].
    ```bash
    python main.py
    ```

---

## 🧠 Model Training

The model uses a regression approach to map hand landmarks to acupressure coordinates.

* [cite_start]**Input:** 42 features (21 landmarks $\times$ 2 coordinates)[cite: 979].
* [cite_start]**Architecture:** Sequential Deep Neural Network with Dense layers (256, 128, 64, 32 units), BatchNormalization, and Dropout [cite: 983-1003].
* [cite_start]**Loss Function:** Mean Squared Error (MSE)[cite: 1029].
* [cite_start]**Optimizer:** Adam[cite: 1028].
* [cite_start]**Performance:** Achieves an average Euclidean Distance Error of 6–10 pixels[cite: 37].

---

## 📊 Results

The system successfully overlays "neon" style indicators on the user's hand in real-time.

* [cite_start]**Status Indicators:** Displays whether the hand is detected and which mode (Cold, Stress, Anxiety) is active[cite: 757].
* [cite_start]**Visual Feedback:** Points are constrained to the hand boundary even during movement[cite: 734].

---

## 🔮 Future Enhancements

* [cite_start]**Dataset Expansion:** Include diverse demographics (age, skin tone) to improve generalization[cite: 322].
* [cite_start]**New Regions:** Extend detection to facial or foot acupressure points using 3D modeling[cite: 323].
* [cite_start]**Mobile Deployment:** Port the system to mobile platforms for better accessibility[cite: 324].
* [cite_start]**AR Integration:** Use Augmented Reality for direct overlay on smartphone cameras[cite: 326].

---

## 👥 Contributors

* [cite_start]**Sachin C** (212222230125) [cite: 4]
* [cite_start]**Bejin B** (212222230021) [cite: 15]
* [cite_start]**Supervisor:** Dr. Karthi Govindharaju [cite: 16]

[cite_start]**Institution:** Saveetha Engineering College, Chennai [cite: 9, 13]
