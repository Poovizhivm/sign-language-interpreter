# Real-Time Indian Sign Language Interpreter Using MobileNetV2

## Live Demo
**Web Application URL**: [https://sign-language-interpreter-nnlw.vercel.app/](https://sign-language-interpreter-nnlw.vercel.app/)

The web interface enables real-time hand gesture recognition directly within modern web browsers via WebAssembly and WebGL acceleration, without requiring a remote Python server.

---

## Overview
This project presents a real-time Indian Sign Language (ISL) interpreter that recognizes hand gestures using a deep learning model based on MobileNetV2 with transfer learning. The system captures hand gestures through a webcam and predicts the corresponding gesture class with low latency and high classification confidence.

---

## Key Features
- **Real-Time Web Inference**: Browser-based gesture recognition powered by ONNX Runtime Web (WASM/WebGL).
- **Desktop Python Pipeline**: Native OpenCV inference script for local workstation environments.
- **Lightweight Architecture**: MobileNetV2 backbone optimized for edge and real-time execution.
- **Transfer Learning**: Custom classification head fine-tuned on hand gesture representations.
- **Gesture Classes**:
  - `hello`
  - `yes`
  - `no`
  - `good`
  - `thank_you`
  - `help`
- **Rigorous Evaluation**: High overall accuracy, precision, recall, F1-score, and ROC-AUC analysis.

---

## Tech Stack
- **Deep Learning**: TensorFlow, Keras, ONNX, ONNX Runtime Web
- **Computer Vision**: OpenCV
- **Scientific Computing**: NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Web Frontend**: HTML5, Vanilla CSS, JavaScript
- **Deployment**: Vercel

---

## Project Structure

```text
sign-language-interpreter/
|
|-- index.html                  # Web application entrypoint
|-- styles.css                  # Responsive application styling
|-- app.js                      # Client-side inference and webcam controller
|-- vercel.json                 # Vercel deployment configuration
|-- .vercelignore               # Vercel build exclusions
|
|-- model/
|   `-- isl_model.onnx          # Web-optimized ONNX model (9.1 MB)
|
|-- models/
|   `-- isl_model.h5            # Trained Keras H5 model (23.7 MB)
|
|-- sample_gestures/            # Sample gesture images for offline testing
|   |-- good.jpg
|   |-- hello.jpg
|   |-- help.jpg
|   |-- no.jpg
|   |-- thank_you.jpg
|   `-- yes.jpg
|
|-- scripts/
|   |-- data_collection.py      # Automated webcam dataset capture tool
|   |-- data_preprocessing.py   # Image loading, normalization and dataset splits
|   |-- train_model.py          # MobileNetV2 fine-tuning and checkpointing
|   |-- evaluate_metrics.py     # Evaluation metrics and figure generation
|   `-- real_time_inference.py  # Native Python OpenCV desktop interpreter
|
|-- data/
|   |-- train/                  # Raw gesture dataset (120 images per class)
|   |-- label_map.pkl           # Label-to-class mapping dictionary
|   |-- X_train.npy, y_train.npy
|   |-- X_val.npy, y_val.npy
|   `-- X_test.npy, y_test.npy
|
|-- images(isl)/                # Methodology and architectural diagrams
|-- requirements.txt            # Python environment dependencies
|-- evaluation_metrics.png      # Nine-panel evaluation report dashboard
|-- roc_auc_curves.png          # One-vs-Rest ROC-AUC curves
|-- confusion_matrix.png        # High-resolution confusion matrix
|-- .gitignore
`-- README.md
```

---

## Methodology

1. **Data Acquisition**
   - Images captured under controlled webcam environments across varying hand positions.
   - 120 images collected per gesture class (720 total images).

2. **Preprocessing Pipeline**
   - Resizing to standard input dimensions (224 x 224 x 3).
   - Dynamic pixel normalization scaled to the range [-1.0, 1.0].
   - Stratified dataset partitioning: 70% training, 15% validation, and 15% testing.
   - Data augmentation during training (rotation, width/height shifts, zoom, shear, and horizontal reflections).

3. **Neural Network Architecture**
   - MobileNetV2 feature extractor initialized with ImageNet weights.
   - GlobalAveragePooling2D dimensionality reduction.
   - Fully-connected dense layers (128 and 64 units) with Batch Normalization, ReLU activations, and Dropout (0.5 and 0.4).
   - 6-unit Softmax classification layer.

4. **Training Strategy**
   - Fine-tuning of the top 30 MobileNetV2 layers.
   - Adam optimizer with an initial learning rate of 1e-4.
   - Categorical cross-entropy loss with balanced class weights.
   - Automated EarlyStopping and ModelCheckpoint monitoring validation accuracy.

---

## Performance Evaluation

Evaluation on the independent test dataset (108 samples) yields the following metrics:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **94.44%** |
| **Precision (macro)** | **0.9530** |
| **Recall (macro)** | **0.9444** |
| **F1-Score (macro)** | **0.9429** |
| **ROC-AUC (macro)** | **1.0000** |

### Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| `good` | 1.0000 | 1.0000 | 1.0000 |
| `hello` | 1.0000 | 0.9444 | 0.9714 |
| `help` | 1.0000 | 1.0000 | 1.0000 |
| `no` | 0.9000 | 1.0000 | 0.9474 |
| `thank_you` | 1.0000 | 0.7222 | 0.8387 |
| `yes` | 0.8182 | 1.0000 | 0.9000 |

---

## Visualizations

### Comprehensive Evaluation Dashboard
![Evaluation Metrics Dashboard](evaluation_metrics.png)

### One-vs-Rest ROC-AUC Curves
![ROC-AUC Curves](roc_auc_curves.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

---

## Execution Guide

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Preprocessing
```bash
python scripts/data_preprocessing.py
```

### 3. Model Training
```bash
python scripts/train_model.py
```

### 4. Metrics Evaluation
```bash
python scripts/evaluate_metrics.py
```

### 5. Desktop Real-Time Inference
```bash
python scripts/real_time_inference.py
```
*Press 'q' in the OpenCV window to terminate inference.*

### 6. Local Web Application
```bash
python -m http.server 8000
```
Open `http://localhost:8000` in any web browser.

---

## Future Scope
- Dataset expansion covering diverse illumination, complex backgrounds, and multiple signers.
- Temporal sequence modeling (LSTM / Transformer) to interpret continuous multi-frame sentence signing.
- Quantization to INT8 format for enhanced mobile execution efficiency via TensorFlow Lite.

---

## Author
**Poovizhi V M** - poovizhivm@gmail.com

---

## License
This project is licensed for academic and research purposes.