import os
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pickle

ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
preprocess_input = keras.applications.mobilenet_v2.preprocess_input
to_categorical = keras.utils.to_categorical

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def preprocess_data(img_size=224, data_dir='data/train'):
    """
    Load, preprocess, and augment image data
    """
    
    images = []
    labels = []
    label_map = {}
    
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        return None
        
    gesture_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"\n{'='*50}")
    print("DATA PREPROCESSING")
    print(f"{'='*50}\n")
    
    # Load all images
    for idx, gesture in enumerate(sorted(gesture_dirs)):
        label_map[gesture] = idx
        gesture_path = os.path.join(data_dir, gesture)
        
        print(f"Loading '{gesture}' images...", end=" ")
        count = 0
        
        for img_file in os.listdir(gesture_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(gesture_path, img_file)
                
                # Read and resize image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                img = preprocess_input(img)
                
                images.append(img)
                labels.append(idx)
                count += 1
        
        print(f"[OK] {count} images")
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\nTotal images: {len(images)}")
    print(f"Shape: {images.shape}\n")
    
    # Split data: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    # Data Augmentation (only for training)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)
    
    # Save preprocessed data
    os.makedirs('data', exist_ok=True)
    np.save('data/X_train.npy', X_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_val.npy', y_val)
    np.save('data/y_test.npy', y_test)
    
    with open('data/label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)
    
    print(f"\n[OK] Data saved successfully!")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_map, datagen

if __name__ == "__main__":
    preprocess_data()