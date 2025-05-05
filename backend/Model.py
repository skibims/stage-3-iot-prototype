import traceback
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from datetime import datetime
import base64
import os
import torch
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
    'SUPABASE_KEY': os.getenv('SUPABASE_KEY'),
    'SUPABASE_BUCKET': os.getenv('SUPABASE_BUCKET'),
    'X_API_KEY': os.getenv('X_API_KEY')
}

# Initialize Supabase client
supabase = create_client(config['SUPABASE_URL'], config['SUPABASE_KEY'])

def extract_plate_position(response_dict):
    """Extract license plate position from API response"""
    try:
        vehicles = response_dict.get("data", {}).get("vehicles", [])
        if vehicles and "plate" in vehicles[0]:
            plate_roi = vehicles[0]["plate"].get("plateROI", {})
            if plate_roi:
                x1 = plate_roi["topLeft"]["x"]
                y1 = plate_roi["topLeft"]["y"]
                x2 = plate_roi["bottomRight"]["x"]
                y2 = plate_roi["bottomRight"]["y"]
                return (int((x1 + x2) / 2), int(y1 - 10))
        return None
    except Exception as e:
        logger.error(f"Error extracting plate position: {e}")
        return None

def draw_detections(frame, detected_motorcycles, license_plate_text, plate_position):
    """Draw motorcycle detections and license plate text"""
    annotated_frame = frame.copy()
    
    # Draw motorcycle detections
    for (*xyxy, conf, cls) in detected_motorcycles:
        x1, y1, x2, y2 = map(int, xyxy)
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add confidence score
        conf_text = f"Motorcycle {conf:.2f}"
        cv2.putText(annotated_frame, conf_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add license plate text at the specified position
    if license_plate_text and plate_position:
        x, y = plate_position
        # Draw white background
        (text_width, text_height), _ = cv2.getTextSize(
            license_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated_frame, 
                     (x - text_width//2 - 5, y - text_height - 5),
                     (x + text_width//2 + 5, y + 5),
                     (255, 255, 255), -1)
        # Draw text
        cv2.putText(annotated_frame, license_plate_text, 
                   (x - text_width//2, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return annotated_frame

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

def send_to_adaptive_api(image_path):
    """Send image to Adaptive Recognition API and return full response"""
    url = "https://api.cloud.adaptiverecognition.com/vehicle/sas"
    headers = {"X-Api-Key": config['X_API_KEY']}
    
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with Image.open(image_path) as img:
            img.save(image_path, 'JPEG', quality=95, optimize=True)
        
        files = {
            "image": ("image.jpg", open(image_path, "rb"), "image/jpeg")
        }
        
        data = {
            "location": "IDN",
            "service": "anpr,mmr"
        }
        
        response = requests.post(url, headers=headers, files=files, data=data, timeout=10)
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        logger.error(f"Error in API call: {e}")
        return None

def upload_to_supabase(file_path, remote_filename):
    """Upload file to Supabase storage using the working method from the first code"""
    try:
        # First verify the image is valid
        test_read = cv2.imread(file_path)
        if test_read is None:
            logger.warning("⚠️ Warning: The saved image may be corrupt")
            return None
        else:
            logger.info(f"✓ Local image verified: {test_read.shape[1]}x{test_read.shape[0]} pixels")
        
        # Read the file as bytes
        with open(file_path, 'rb') as file:
            file_bytes = file.read()
        
        # Upload using supabase storage client directly
        storage_response = supabase.storage.from_(config['SUPABASE_BUCKET']).upload(
            path=remote_filename,  # Remote path/filename in bucket
            file=file_bytes,       # File content as bytes
            file_options={"content-type": "image/jpeg"}
        )
        
        # Get the public URL for the uploaded file
        public_url = supabase.storage.from_(config['SUPABASE_BUCKET']).get_public_url(remote_filename)
        
        logger.info(f"✅ Image uploaded to Supabase storage as: {remote_filename}")
        logger.info(f"🔗 Public URL: {public_url}")
        
        return public_url
        
    except Exception as upload_error:
        logger.error(f"❌ Supabase storage upload error: {upload_error}")
        traceback_str = traceback.format_exc()
        logger.error(f"Stack trace: {traceback_str}")
        
        # Try alternative method with explicit encoding
        try:
            logger.info("🔄 Trying alternative upload method...")
            
            # Read the image and ensure proper JPEG encoding
            img = cv2.imread(file_path)
            is_success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not is_success:
                logger.error("⚠️ Error encoding image to JPEG")
                return None
                
            # Convert buffer to bytes
            file_bytes = buffer.tobytes()
            
            # Upload using upsert method (replaces if exists)
            storage_response = supabase.storage.from_(config['SUPABASE_BUCKET']).upload(
                path=remote_filename,
                file=file_bytes,
                file_options={
                    "content-type": "image/jpeg",
                    "upsert": True
                }
            )
            
            logger.info(f"✅ Image uploaded via alternative method: {remote_filename}")
            
            # Get the public URL
            public_url = supabase.storage.from_(config['SUPABASE_BUCKET']).get_public_url(remote_filename)
            return public_url
            
        except Exception as alt_error:
            logger.error(f"❌ Alternative upload also failed: {alt_error}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return None

# Initialize Flask and YOLO model
app = Flask(__name__)
motorcycle_model = load_yolo_model()

@app.route('/upload', methods=['POST'])
def classify_image():
    logger.info("📥 Received new request at /upload")
    
    try:
        # Get device_id from request
        device_id = "unknown"
        
        # Process incoming image
        if 'image' in request.files:
            image = request.files['image']
            device_id = request.form.get("device_id", "unknown")
            if not image.filename:
                return jsonify({"error": "No selected file"}), 400
                
            in_memory_file = BytesIO()
            image.save(in_memory_file)
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
        elif request.is_json:
            data = request.get_json()
            device_id = data.get("device_id", "unknown")
            image_base64 = data.get("image_base64")
            if not image_base64:
                return jsonify({"error": "Missing base64 image"}), 400
                
            img_data = base64.b64decode(image_base64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            return jsonify({"error": "Unsupported Content-Type"}), 415

        # Create directory for saved images if it doesn't exist
        saved_dir = "saved_images"
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        # Save temporary image for API
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, frame)

        # Get API response and extract information
        api_response = send_to_adaptive_api(temp_image_path)
        plate_position = None
        license_plate_text = None
        
        if api_response:
            plate_position = extract_plate_position(api_response)
            vehicles = api_response.get("data", {}).get("vehicles", [])
            if vehicles and "plate" in vehicles[0] and vehicles[0]["plate"]["found"]:
                license_plate_text = vehicles[0]["plate"].get("separatedText", "N/A")
                logger.info(f"🔍 License Plate Result: {license_plate_text}")
            else:
                logger.info("No license plate detected")

        # Detect motorcycles
        results = motorcycle_model(frame)
        detections = results.xyxy[0].to('cpu')
        detected_motorcycles = [d for d in detections if int(d[5]) == 3]

        if detected_motorcycles:
            highest_conf = max([float(d[4]) for d in detected_motorcycles])
            logger.info(f"🚨 Detected {len(detected_motorcycles)} motorcycles. Highest confidence: {highest_conf:.2f}")
            
            # Draw detections and save image
            annotated_frame = draw_detections(
                frame, 
                detected_motorcycles, 
                license_plate_text, 
                plate_position
            )
            
            # Generate filename with device_id and timestamp
            output_filename = f"{device_id}_detection_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            output_path = os.path.join(saved_dir, output_filename)
            cv2.imwrite(output_path, annotated_frame)

            # Upload to Supabase using the working method
            public_url = upload_to_supabase(output_path, output_filename)
            
            # Delete temporary files
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            return jsonify({
                "status": "success",
                "result": "motorcycle",
                "motorcycle_detections": len(detected_motorcycles),
                "license_plate_text": license_plate_text if license_plate_text else "N/A",
                "confidence": round(highest_conf, 2),
                "filename": output_filename,
                "image_url": public_url
            }), 200
        
        else:
            logger.info("✅ No motorcycles detected.")
            
            # If we have a license plate but no motorcycle (possible false negative),
            # save the image anyway with the license plate text
            if license_plate_text:
                # Add license plate text to the frame
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, f"Plate: {license_plate_text}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Save annotated frame
                plate_filename = f"{device_id}_plate_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                plate_path = os.path.join(saved_dir, plate_filename)
                cv2.imwrite(plate_path, annotated_frame)
                
                # Upload to Supabase
                public_url = upload_to_supabase(plate_path, plate_filename)
                
                # Delete temporary files
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                
                return jsonify({
                    "status": "success",
                    "result": "license_plate_only",
                    "motorcycle_detections": 0,
                    "license_plate_text": license_plate_text,
                    "filename": plate_filename,
                    "image_url": public_url
                }), 200
            else:
                # Delete temporary files
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                    
                return jsonify({
                    "status": "success",
                    "result": "none",
                    "motorcycle_detections": 0,
                    "license_plate_text": "N/A"
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