import cv2
import os
import sys
from datetime import datetime

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def collect_gesture_data(gesture_name, num_images=120, camera_index=0):
    """
    Collect images for a specific gesture using webcam
    
    Args:
        gesture_name: Name of the gesture (e.g., 'hello')
        num_images: Number of images to collect (default: 120)
        camera_index: Index of camera device (default: 0)
    """
    
    # Create directory if it doesn't exist
    data_dir = f'data/train/{gesture_name}'
    os.makedirs(data_dir, exist_ok=True)
    
    # Open webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not access camera {camera_index}.")
        print("Please ensure a webcam is connected and accessible.")
        return
        
    count = 0
    
    print(f"\n{'='*50}")
    print(f"Collecting images for gesture: {gesture_name.upper()}")
    print(f"{'='*50}")
    print(f"Press 'c' to capture, 'q' to quit")
    print(f"Target: {num_images} images\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nError reading camera feed.")
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Display instructions
        cv2.putText(frame, f'Gesture: {gesture_name.upper()}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Images: {count}/{num_images}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, 'Press C to Capture | Q to Quit', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        
        cv2.imshow('Gesture Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f'{data_dir}/{gesture_name}_{count:03d}.jpg'
            cv2.imwrite(filename, frame)
            count += 1
            print(f"[OK] Captured: {filename}")
            
            if count >= num_images:
                print(f"\n[OK] Successfully collected {num_images} images!")
                break
        
        elif key == ord('q'):
            print(f"[STOP] Collection stopped. {count} images saved.")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Collect data for each gesture
    gestures = ['hello', 'thank_you', 'yes', 'no', 'help', 'good']
    
    for gesture in gestures:
        collect_gesture_data(gesture, num_images=120)
        input(f"\nPress Enter to continue to next gesture...")