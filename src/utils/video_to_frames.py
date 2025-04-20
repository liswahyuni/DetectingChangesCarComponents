import cv2
import os
from datetime import datetime

class VideoFrameExtractor:
    def __init__(self):
        self.data_dir = "/Users/liswahyuni/Documents/PrincipalAIEngineer_SMARTM2M_Indonesia/dataset"
        self.frames_dir = os.path.join(self.data_dir, "video_frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        
    def extract_frames(self, video_path, frame_interval=1):
        """
        Extract frames from video
        Args:
            video_path: Path to video file
            frame_interval: Extract frame every N frames (default=1 means every frame)
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Video properties:")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Save frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                frame_filename = f"frame_{timestamp}.jpg"
                frame_path = os.path.join(self.frames_dir, frame_filename)
                
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                
                if saved_count % 100 == 0:
                    print(f"Saved {saved_count} frames...")
            
            frame_count += 1
        
        cap.release()
        print(f"\nExtraction completed:")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames saved: {saved_count}")
        print(f"Frames saved to: {self.frames_dir}")

if __name__ == "__main__":
    extractor = VideoFrameExtractor()
    
    # Replace with your video path
    video_path = "/Users/liswahyuni/Documents/PrincipalAIEngineer_SMARTM2M_Indonesia/dataset/car_recording.mp4"
    
    # Extract every frame (adjust frame_interval if you want to skip frames)
    extractor.extract_frames(video_path, frame_interval=1)