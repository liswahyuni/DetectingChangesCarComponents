# Car Component Detection System

A deep learning system that detects and reports the status of car components (doors and hood) in real-time from a 3D web interface.

## Features

- Real-time detection of car component states (open/closed)
- Web interface for visualizing component status
- Natural language descriptions of car state
- Lightweight model optimized for real-time inference

## Requirements

- Python 3.8+
- TensorFlow 2.16+
- Flask 3.0+
- Chrome/Chromium browser (for Selenium)
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository:
```bash
git clone https://github.com/liswahyuni/DetectingChangesCarComponents.git

cd DetectingChangesCarComponents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Start the web server:
```bash
python src/app.py
```

Open your browser and navigate to:
```bash
http://localhost:8000
```

3. The interface will automatically refresh to show the current state of car components.
## Project Structure
sDetectingChangesCarComponents/
├── dataset/                # Dataset directory
├── src/
│   ├── app.py             # Main Flask application
│   ├── train_detector.py  # Training script
│   ├── models/
│   │   ├── car_detector.py          # Component detection model
│   │   ├── vision_language_model.py # Description generation
│   │   └── visual_grounding_model.py # Visual grounding
│   ├── templates/
│   │   ├── car_status.html         # Main status page
│   │   ├── door_description.html   # Description page
│   │   └── visual_grounding.html   # Grounding page
│   └── utils/
│       ├── generate_labels.py      # Label generation
│       ├── video_processor.py      # Video processing
│       └── video_to_frames.py      # Frame extraction
├── README.md
└── requirements.txt

## Model Architecture
The system uses a lightweight convolutional neural network based on MobileNetV2 architecture for efficient real-time inference. The model processes 224x224 RGB images and outputs the probability of each component being open or closed.

## Troubleshooting
### Common Issues
1. Selenium WebDriver Error :
   
   - Make sure Chrome/Chromium is installed
   - Check that the Chrome version matches the chromedriver version
2. Model Prediction Issues :
   
   - Ensure the web interface is accessible
   - Check network connectivity to the 3D car model web view
3. Flask Server Not Starting :
   
   - Verify port 8000 is not in use
   - Check for proper permissions

## Future Improvements
- Add support for more car components
- Improve model accuracy with larger training dataset
- Implement user authentication
- Add mobile responsiveness