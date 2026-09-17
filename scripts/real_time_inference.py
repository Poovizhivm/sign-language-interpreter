import os
import sys
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow import keras
preprocess_input = keras.applications.mobilenet_v2.preprocess_input
import time

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

class ISLInterpreter:
    def __init__(self, model_path='models/isl_model.h5', 
                 label_map_path='data/label_map.pkl'):
        """Initialize the ISL interpreter"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'. Please train the model first using scripts/train_model.py.")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"Label map not found at '{label_map_path}'. Please run scripts/data_preprocessing.py first.")
            
        print("Loading model...", end=" ")
        self.model = load_model(model_path)
        print("[OK]")
        
        print("Loading label map...", end=" ")
        with open(label_map_path, 'rb') as f:
            self.label_map = pickle.load(f)
        print("[OK]")
        
        # Reverse label map (index -> gesture name)
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        self.img_size = 224
        self.confidence_threshold = 0.6
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        
        # Resize
        resized = cv2.resize(frame, (self.img_size, self.img_size))
        
        # Normalize
        normalized = preprocess_input(resized)
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def predict_gesture(self, frame):
        """Predict gesture from frame"""
        
        # Preprocess
        processed = self.preprocess_frame(frame)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get top prediction
        pred_index = np.argmax(predictions)
        confidence = predictions[pred_index]
        gesture_name = self.reverse_label_map[pred_index]
        
        return gesture_name, confidence, predictions
    
    def run_inference(self, camera_index=0):
        """Run real-time inference from webcam"""
        
        print(f"\n{'='*50}")
        print("REAL-TIME ISL INTERPRETER")
        print(f"{'='*50}")
        print("\nGestures:", list(self.label_map.keys()))
        print("Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera with index {camera_index}.")
            print("Please ensure a webcam is connected and not being used by another application.")
            return
            
        fps_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nCamera stream ended or disconnected.")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convert BGR to RGB for model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict
            gesture, confidence, all_predictions = self.predict_gesture(frame_rgb)
            
            # Draw predictions
            if confidence >= self.confidence_threshold:
                cv2.rectangle(frame, (10, 10), (400, 100), (0, 255, 0), -1)
                cv2.putText(frame, f"Gesture: {gesture.upper()}", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                cv2.putText(frame, f"Confidence: {confidence:.2%}", (30, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 255), -1)
                cv2.putText(frame, "Low Confidence", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(frame, f"{confidence:.2%}", (30, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display all confidence scores
            y_offset = 120
            for idx, gesture_name in enumerate(self.label_map.keys()):
                score = all_predictions[self.label_map[gesture_name]]
                color = (0, 255, 0) if gesture_name == gesture else (200, 200, 200)
                cv2.putText(frame, f"{gesture_name}: {score:.2%}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y_offset += 30
            
            # FPS counter
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_time)
                fps_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (w-150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('ISL Interpreter', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    interpreter = ISLInterpreter()
    interpreter.run_inference()