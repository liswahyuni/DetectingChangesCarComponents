from transformers import pipeline, DetrImageProcessor, logging
import torch
import cv2
import numpy as np
import warnings
import os

class CarVisualGroundingModel:
    def __init__(self):
        # Configure logging and warnings
        logging.set_verbosity_error()
        warnings.filterwarnings('ignore')
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        
        # Initialize model
        self.model = pipeline(
            "object-detection",
            model="facebook/detr-resnet-50",
            device="cpu"  # Force CPU for better compatibility
        )
        
        # Define car components and their aliases
        self.car_components = {
            'door': ['front left door', 'front right door', 'rear left door', 'rear right door'],
            'hood': ['hood', 'bonnet', 'front hood'],
            'trunk': ['trunk', 'boot', 'rear trunk']
        }
        
        # Define colors for visualization
        self.colors = {
            'door': (0, 255, 0),    # Green
            'hood': (255, 0, 0),    # Blue
            'trunk': (0, 0, 255)    # Red
        }

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if image is None:
            raise ValueError("Input image is None")
            
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        return cv2.resize(image, (800, 600))  # Resize for consistent processing

    def draw_box_with_label(self, image, box, label, color):
        """Draw bounding box and label on image"""
        # Draw box
        cv2.rectangle(
            image,
            (int(box['xmin']), int(box['ymin'])),
            (int(box['xmax']), int(box['ymax'])),
            color,
            2
        )
        
        # Calculate label size and position
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        label_ymin = max(int(box['ymin'] - 10), label_size[1] + 10)
        
        # Draw label background
        cv2.rectangle(
            image,
            (int(box['xmin']), label_ymin - label_size[1] - 10),
            (int(box['xmin'] + label_size[0]), label_ymin),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (int(box['xmin']), label_ymin),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    def locate_objects(self, image, text_prompt):
        """Locate and visualize car components based on text prompt"""
        try:
            # Input validation
            if not text_prompt:
                raise ValueError("Text prompt is empty")
                
            # Preprocess image
            image = self.preprocess_image(image)
            img_with_boxes = image.copy()
            text_prompt = text_prompt.lower()
            
            # Get predictions
            predictions = self.model(image)
            
            found_objects = False
            for pred in predictions:
                if pred['score'] < 0.5:
                    continue
                    
                # Match component type
                component_type = None
                for type_key, aliases in self.car_components.items():
                    if any(alias in text_prompt for alias in aliases):
                        component_type = type_key
                        break
                
                if component_type:
                    label = f"{component_type.upper()} ({pred['score']:.2f})"
                    self.draw_box_with_label(
                        img_with_boxes, 
                        pred['box'], 
                        label, 
                        self.colors[component_type]
                    )
                    found_objects = True
            
            if not found_objects:
                cv2.putText(
                    img_with_boxes,
                    "No matching components found",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            
            return cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            # Return original image with error message
            cv2.putText(
                image,
                f"Error: {str(e)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)