import os
import sys
import numpy as np
import pickle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def build_model(num_classes=6, img_size=224):
    """
    Build CNN model using Transfer Learning (MobileNetV2)
    """
    
    print(f"\n{'='*50}")
    print("BUILDING MODEL")
    print(f"{'='*50}\n")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    return model

def train_model(epochs=50, batch_size=32):
    """
    Train the model with preprocessed data
    """
    os.makedirs('models', exist_ok=True)
    
    # Load preprocessed data
    X_train = np.load('data/X_train.npy')
    X_val = np.load('data/X_val.npy')
    y_train = np.load('data/y_train.npy')
    y_val = np.load('data/y_val.npy')
    
    y_integers = np.argmax(y_train, axis=1)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weights = dict(enumerate(class_weights))
    print("Class Weights:", class_weights)
    
    # Build model
    model = build_model(num_classes=y_train.shape[1])
    base_model = model.layers[0]

    # Unfreeze last 30 layers
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    # Recompile model to reflect unfrozen layers
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/isl_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print(f"\n{'='*50}")
    print("TRAINING MODEL")
    print(f"{'='*50}\n")
    
    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\n{'='*50}")
    print("MODEL EVALUATION")
    print(f"{'='*50}\n")
    
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()