from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import cv2
import numpy as np

class CarVisionLanguageModel:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = cv2.resize(image, (384, 384))
        return image

    def describe_image(self, image):
        try:
            processed_image = self.preprocess_image(image)
            
            with torch.no_grad():
                pixel_values = self.feature_extractor(processed_image, return_tensors="pt").pixel_values.to(self.device)
                
                prompt = "Describe the current state of the car's doors and hood in detail:"
                prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                output_ids = self.model.generate(
                    pixel_values,
                    max_length=150,
                    num_beams=5,
                    length_penalty=1.0,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                )
                
                description = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                return self._format_description(description)
                
        except Exception as e:
            return "Unable to analyze the car's current state."

    def _format_description(self, description):
        # Extract component states
        components = {
            'front left door': 'closed',
            'front right door': 'closed',
            'rear left door': 'closed',
            'rear right door': 'closed',
            'hood': 'closed'
        }
        
        desc_lower = description.lower()
        
        # Analyze text for component states
        for component in components:
            if component in desc_lower:
                if any(state in desc_lower for state in ['open', 'opened', 'opening']):
                    components[component] = 'open'
                    
        # Format final description
        result = "The car's "
        open_components = [comp for comp, state in components.items() if state == 'open']
        closed_components = [comp for comp, state in components.items() if state == 'closed']
        
        if open_components:
            result += f"{', '.join(open_components[:-1])}"
            if len(open_components) > 1:
                result += f" and {open_components[-1]}"
            elif len(open_components) == 1:
                result += f"{open_components[0]}"
            result += " is open" if len(open_components) == 1 else " are open"
        
        if open_components and closed_components:
            result += ", while "
        
        if closed_components:
            result += f"the {', '.join(closed_components[:-1])}"
            if len(closed_components) > 1:
                result += f" and {closed_components[-1]}"
            elif len(closed_components) == 1:
                result += f"{closed_components[0]}"
            result += " remains closed" if len(closed_components) == 1 else " remain closed"
        
        return result + "."