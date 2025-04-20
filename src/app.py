import tensorflow as tf
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import cv2
import numpy as np
from models.car_detector import CarComponentDetector
from models.vision_language_model import CarVisionLanguageModel
from models.visual_grounding_model import CarVisualGroundingModel
import time
import base64
import os
import warnings
import logging

# Configure TensorFlow for M1/M2 compatibility
tf.keras.optimizers.Adam = tf.keras.optimizers.legacy.Adam

# Configure logging and warnings
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

app = Flask(__name__)
CORS(app)

# Initialize models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    detector = CarComponentDetector()
    vision_model = CarVisionLanguageModel()
    grounding_model = CarVisualGroundingModel()

def setup_browser():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--single-process')
    return webdriver.Chrome(options=chrome_options)

@app.route('/')
def index():
    return render_template('car_status.html')

@app.route('/door_description')
def door_description():
    return render_template('door_description.html')

@app.route('/visual_grounding')
def visual_grounding():
    return render_template('visual_grounding.html')

@app.route('/get_car_status')
def get_car_status():
    try:
        browser = setup_browser()
        browser.get('http://103.154.152.215:11234/')
        time.sleep(2)
        
        screenshot = browser.get_screenshot_as_png()
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        
        predictions = detector.predict(img)
        
        formatted_predictions = {
            component.split()[1] if 'Front' in component else component.split()[-1]: 
            'CLOSED' if status == 'Closed' else 'OPEN'
            for component, status in predictions.items()
        }
        
        return jsonify(formatted_predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'browser' in locals():
            browser.quit()

@app.route('/describe_car')
def describe_car():
    try:
        browser = setup_browser()
        browser.get('http://103.154.152.215:11234/')
        time.sleep(1)
        
        screenshot = browser.get_screenshot_as_png()
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        
        description = vision_model.describe_image(img).replace("Car Status Description: ", "")
        
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jsonify({
            "description": description,
            "image": base64.b64encode(buffer.tobytes()).decode('utf-8')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'browser' in locals():
            browser.quit()

@app.route('/ground_objects', methods=['POST'])
def ground_objects():
    try:
        browser = setup_browser()
        browser.get('http://103.154.152.215:11234/')
        time.sleep(1)
        
        screenshot = browser.get_screenshot_as_png()
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        
        text_prompt = request.json.get('prompt', '')
        if not any(word in text_prompt.lower() for word in ['door', 'hood', 'trunk', 'window', 'wheel']):
            text_prompt = f"Car {text_prompt}".strip()
        
        result_img = grounding_model.locate_objects(img, text_prompt)
        
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jsonify({
            "image": base64.b64encode(buffer.tobytes()).decode('utf-8'),
            "prompt": text_prompt
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'browser' in locals():
            browser.quit()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)