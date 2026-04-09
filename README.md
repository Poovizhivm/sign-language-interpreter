# Real-Time Indian Sign Language Interpreter Using MobileNetV2

## Overview
This project presents a real-time Indian Sign Language (ISL) interpreter that recognizes hand gestures using a deep learning model based on MobileNetV2 with transfer learning. The system captures hand gestures through a webcam and predicts the corresponding gesture class in real time.

---

## Features
- Real-time gesture recognition using webcam
- Lightweight and efficient MobileNetV2 architecture
- Transfer learning for improved performance
- Six gesture classes:
  - hello
  - yes
  - no
  - good
  - thank_you
  - help

---

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Project Structure


sign-language-interpreter/
│
├── data/
│ ├── train/ # Dataset (excluded from repo)
│ └── label_map.pkl
│
├── models/
│ └── isl_model.h5 # Trained model (optional)
│
├── scripts/
│ ├── data_collection.py
│ ├── data_preprocessing.py
│ ├── train_model.py
│ ├── evaluate_metrics.py
│ └── real_time_inference.py
│
├── evaluation_metrics.png
├── .gitignore
└── README.md


---

## Methodology

1. **Data Collection**
   - Captured hand gesture images using webcam
   - 120 images per class

2. **Preprocessing**
   - Resizing images to model input size
   - Normalization
   - Train-validation-test split

3. **Model**
   - MobileNetV2 (pre-trained on ImageNet)
   - Custom classification head
   - Softmax output layer

4. **Training**
   - Transfer learning approach
   - Fine-tuning for gesture classification

5. **Inference**
   - Real-time prediction using webcam feed

---

## How to Run

### 1. Preprocess Data
```bash
python scripts/data_preprocessing.py
2. Train Model
python scripts/train_model.py
3. Evaluate Model
python scripts/evaluate_metrics.py
4. Run Real-Time Inference
python scripts/real_time_inference.py
Results
Accuracy: 94.44%
Precision: 0.9530
Recall: 0.9444
F1-Score: 0.9429

Evaluation metrics visualization:

Dataset

The dataset is not included in this repository due to size constraints.

Images were collected manually using a webcam
Each class contains approximately 120 images
Future Improvements
Increase dataset size for better generalization
Add more gesture classes
Improve robustness to lighting/background variations
Deploy as a web/mobile application

Author
Poovizhi V M - poovizhivm@gmail.com

License
This project is for academic purposes.