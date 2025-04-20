import cv2
import os
import pandas as pd
from datetime import datetime

class VideoProcessor:
    def __init__(self):
        self.data_dir = "/Users/liswahyuni/Documents/PrincipalAIEngineer_SMARTM2M_Indonesia/dataset"
        self.videos_dir = os.path.join(self.data_dir, "videos")
        self.frames_dir = os.path.join(self.data_dir, "video_frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        
        self.components = {
            'front_left': 'Front_Left_Door',
            'front_right': 'Front_Right_Door',
            'rear_left': 'Rear_Left_Door',
            'rear_right': 'Rear_Right_Door',
            'hood': 'Hood'
        }

    def process_videos(self):
        data = []
        
        # Process each video file
        print(os.listdir(self.videos_dir))
        for video_file in os.listdir(self.videos_dir):
            if not video_file.endswith('.mp4'):
                continue
                
            print(f"Processing video: {video_file}")
            video_path = os.path.join(self.videos_dir, video_file)
            
            # Extract active components from filename
            active_components = video_file.replace('.mp4', '').split('_')
            
            # Process video frames
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame every 3 frames to reduce redundancy
                if frame_count % 3 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    frame_filename = f"frame_{video_file}_{timestamp}.jpg"
                    frame_path = os.path.join(self.frames_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    
                    # Create labels based on video filename
                    labels = {
                        'Front_Left_Door': 1 if 'front_left' in active_components else 0,
                        'Front_Right_Door': 1 if 'front_right' in active_components else 0,
                        'Rear_Left_Door': 1 if 'rear_left' in active_components else 0,
                        'Rear_Right_Door': 1 if 'rear_right' in active_components else 0,
                        'Hood': 1 if 'hood' in active_components else 0
                    }
                    
                    data.append({
                        'image_filename': frame_filename,
                        'timestamp': timestamp,
                        'source_video': video_file,
                        **labels
                    })
                
                frame_count += 1
            
            cap.release()
            print(f"Extracted {frame_count} frames from {video_file}")
        
        # Save labels to CSV
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.data_dir, "frame_labels.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nProcessing completed!")
        print(f"Total frames extracted: {len(data)}")
        print(f"Labels saved to: {csv_path}")

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_videos()