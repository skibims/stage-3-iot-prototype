import traceback
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from datetime import datetime
import base64
import os
import torch
import boto3
import cv2
import numpy as np
from io import BytesIO
from supabase import create_client
import requests
import json
import logging
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
config = {
    'SUPABASE_URL': os.getenv('SUPABASE_URL'),
    'SUPABASE_S3_ENDPOINT': f"{os.getenv('SUPABASE_URL')}/storage/v1/s3",
    'SUPABASE_REGION': os.getenv('SUPABASE_REGION'),
    'ACCESS_KEY_ID': os.getenv('ACCESS_KEY_ID'),
    'SECRET_ACCESS_KEY': os.getenv('SECRET_ACCESS_KEY'),
    'SUPABASE_BUCKET': os.getenv('SUPABASE_BUCKET'),
    'SUPABASE_KEY': os.getenv('SUPABASE_KEY'),
    'X_API_KEY': os.getenv('X_API_KEY')
}

# Initialize clients
supabase = create_client(config['SUPABASE_URL'], config['SUPABASE_KEY'])
s3 = boto3.client(
    's3',
    aws_access_key_id=config['ACCESS_KEY_ID'],
    aws_secret_access_key=config['SECRET_ACCESS_KEY'],
    endpoint_url=config['SUPABASE_S3_ENDPOINT'],
    region_name=config['SUPABASE_REGION'],
)

def draw_detection(frame, xyxy, license_plate_text, confidence):
    """Draw detection box and license plate text on image."""
    x1, y1, x2, y2 = map(int, xyxy)
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Prepare text
    text = f"Motorcycle {confidence:.2f}"
    if license_plate_text:
        text += f" | Plate: {license_plate_text}"
    
    # Draw text background
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 255, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def load_yolo_model():
    """Initialize and load YOLO model with specific configurations."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov11.pt')
        model.to(device)
        model.conf = 0.25
        model.iou = 0.45
        model.classes = [3]  # Motorcycle only
        logger.info(f"YOLO model loaded successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        raise

def preprocess_image(frame):
    """Preprocess image for better detection."""
    try:
        # Resize image while maintaining aspect ratio
        max_size = 1024
        height, width = frame.shape[:2]
        scale = min(max_size/width, max_size/height)
        new_size = (int(width*scale), int(height*scale))
        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        
        # Enhance image
        enhanced = cv2.convertScaleAbs(resized, alpha=1.2, beta=0)
        return enhanced
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def send_to_adaptive_api(image_path):
    """Send image to Adaptive Recognition API with improved error handling."""
    url = "https://api.cloud.adaptiverecognition.com/vehicle/sas"
    headers = {"X-Api-Key": config['X_API_KEY']}
    
    try:
        # Ensure image exists and is readable
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Optimize image before sending
        with Image.open(image_path) as img:
            # Save with optimal quality
            img.save(image_path, 'JPEG', quality=95, optimize=True)
        
        multipart_data = {
            "image": ("image.jpg", open(image_path, "rb"), "image/jpeg"),
            "location": (None, "IDN"),
            "service": (None, "anpr,mmr")
        }
        
        response = requests.post(url, headers=headers, files=multipart_data, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        
        response_dict = response.json()
        logger.info(f"API Response: {response_dict}")
        
        vehicles = response_dict.get("data", {}).get("vehicles", [])
        if vehicles and "plate" in vehicles[0] and vehicles[0]["plate"]["found"]:
            return vehicles[0]["plate"].get("separatedText", "N/A")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in API call: {e}")
        return None

def save_annotated_image(frame, license_plate_text, detected_motorcycles):
    """Save image with annotations for motorcycle detections and license plate."""
    try:
        annotated_frame = frame.copy()
        detection_results = []

        for idx, (*xyxy, conf, cls) in enumerate(detected_motorcycles):
            # Draw detection on image
            annotated_frame = draw_detection(annotated_frame, xyxy, license_plate_text, conf)
            
            # Save detection info
            x1, y1, x2, y2 = map(int, xyxy)
            detection_results.append({
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2]
            })

        # Save annotated image
        output_filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(output_filename, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Saved annotated image: {output_filename}")

        return output_filename, detection_results
    except Exception as e:
        logger.error(f"Error saving annotated image: {e}")
        raise

# Initialize Flask and YOLO model
app = Flask(__name__)
motorcycle_model = load_yolo_model()

@app.route('/upload', methods=['POST'])
def classify_image():
    logger.info("📥 Received new request at /upload")
    
    try:
        # Process incoming image
        frame = None
        if 'image' in request.files:
            image = request.files['image']
            if not image.filename:
                return jsonify({"error": "No selected file"}), 400
                
            in_memory_file = BytesIO()
            image.save(in_memory_file)
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
        elif request.is_json:
            data = request.get_json()
            image_base64 = data.get("image_base64")
            if not image_base64:
                return jsonify({"error": "Missing base64 image"}), 400
                
            img_data = base64.b64decode(image_base64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            return jsonify({"error": "Unsupported Content-Type"}), 415

        # Preprocess image
        frame = preprocess_image(frame)
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Get license plate from API
        license_plate_text = send_to_adaptive_api(temp_image_path)
        logger.info(f"🔍 License Plate: {license_plate_text if license_plate_text else 'Not detected'}")

        # Detect motorcycles
        results = motorcycle_model(frame)
        detections = results.xyxy[0].to('cpu')
        detected_motorcycles = [d for d in detections if int(d[5]) == 3]

        if detected_motorcycles:
            highest_conf = max([float(d[4]) for d in detected_motorcycles])
            logger.info(f"🚨 Motorcycle detected! Confidence: {highest_conf:.2f}")

            # Save annotated image
            output_filename, detection_results = save_annotated_image(
                frame, license_plate_text, detected_motorcycles
            )

            return jsonify({
                "status": "success",
                "result": "motorcycle_and_license_plate",
                "motorcycle_detections": {
                    "count": len(detected_motorcycles),
                    "details": detection_results
                },
                "license_plate_text": license_plate_text.strip() if license_plate_text else "N/A",
                "output_image": output_filename
            }), 200

        return jsonify({
            "status": "success",
            "result": "license_plate_only" if license_plate_text else "none",
            "license_plate_text": license_plate_text.strip() if license_plate_text else "N/A"
        }), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)