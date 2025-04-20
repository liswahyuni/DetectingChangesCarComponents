import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from models.car_detector import CarComponentDetector

def load_dataset(data_dir, labels_file):
    # Read labels
    df = pd.read_csv(labels_file)
    
    # Load images and labels
    images = []
    labels = []
    
    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, 'video_frames', row['image_filename'])
        img = cv2.imread(img_path)
        if img is not None:
            # Match model input specs
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            
            # Binary labels with updated order
            label = [
                row['Front_Left_Door'],
                row['Front_Right_Door'],
                row['Rear_Left_Door'],
                row['Rear_Right_Door'],
                row['Hood']
            ]
            labels.append(label)
    
    return np.array(images), np.array(labels)

def main():
    # Load dataset
    data_dir = "/Users/liswahyuni/Documents/PrincipalAIEngineer_SMARTM2M_Indonesia/dataset"
    labels_file = os.path.join(data_dir, "frame_labels.csv")
    
    images, labels = load_dataset(data_dir, labels_file)
    
    # Add data augmentation and validation split
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, 
        test_size=0.2, 
        random_state=42,
        shuffle=True  # Added shuffle for better generalization
    )
    
    # Initialize model with training configuration
    detector = CarComponentDetector()
    
    # Train with callbacks
    history = detector.train(
        train_images, 
        train_labels,
        val_images,
        val_labels,
        epochs=30,
        batch_size=32
    )
    
    # Save in modern format
    detector.model.save('car_detector_model.keras')
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()