import os
import pandas as pd
from datetime import datetime

class FrameLabeler:
    def __init__(self):
        self.data_dir = "/Users/liswahyuni/Documents/PrincipalAIEngineer_SMARTM2M_Indonesia/dataset"
        self.frames_dir = os.path.join(self.data_dir, "video_frames")
        self.csv_path = os.path.join(self.data_dir, "frame_labels.csv")
        
        self.classes = [
            'Front_Left_Door',
            'Front_Right_Door',
            'Rear_Left_Door',
            'Rear_Right_Door',
            'Hood'
        ]
        
        # Define ranges for each component
        self.ranges = {
            'Hood': (1, 350),
            'Rear_Right_Door': (351, 776),
            'Rear_Left_Door': (777, 1103),
            'Front_Right_Door': (1104, 1442),
            'Front_Left_Door': (1442, 1772)
        }

    def auto_label(self):
        frame_files = [f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')]
        frame_files.sort()
        
        data = []
        
        for idx, frame_file in enumerate(frame_files, 1):
            # First, set labels based on ranges
            labels = {
                'image_filename': frame_file,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'Front_Left_Door': 1 if self.ranges['Front_Left_Door'][0] <= idx <= self.ranges['Front_Left_Door'][1] else 0,
                'Front_Right_Door': 1 if self.ranges['Front_Right_Door'][0] <= idx <= self.ranges['Front_Right_Door'][1] else 0,
                'Rear_Left_Door': 1 if self.ranges['Rear_Left_Door'][0] <= idx <= self.ranges['Rear_Left_Door'][1] else 0,
                'Rear_Right_Door': 1 if self.ranges['Rear_Right_Door'][0] <= idx <= self.ranges['Rear_Right_Door'][1] else 0,
                'Hood': 1 if self.ranges['Hood'][0] <= idx <= self.ranges['Hood'][1] else 0
            }
            
            # Get components from filename
            filename_parts = frame_file.split('frame_')[1].split('_2024')[0]  # Remove timestamp
            filename_parts = filename_parts.replace('.mp4', '')
            
            # Set Hood if present
            if 'hood' in filename_parts:
                labels['Hood'] = 1
            
            # Check for door components
            if 'front_left' in filename_parts:
                labels['Front_Left_Door'] = 1
            if 'front_right' in filename_parts:
                labels['Front_Right_Door'] = 1
            if 'rear_left' in filename_parts:
                labels['Rear_Left_Door'] = 1
            if 'rear_right' in filename_parts:
                labels['Rear_Right_Door'] = 1
            
            data.append(labels)
            
            if len(data) % 100 == 0:
                print(f"Processed {len(data)} images...")
        
        # Save all labels
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)
        print(f"\nLabeling completed!")
        print(f"Total frames labeled: {len(data)}")
        print(f"Labels saved to: {self.csv_path}")

if __name__ == "__main__":
    labeler = FrameLabeler()
    labeler.auto_label()