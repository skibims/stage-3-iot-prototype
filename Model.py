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
import requests  # Import requests for API calls
import json  # Import json for parsing API responses

# Load environment
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_S3_ENDPOINT = f"{SUPABASE_URL}/storage/v1/s3"
SUPABASE_REGION = os.getenv("SUPABASE_REGION")
ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
X_API_KEY = os.getenv("X_API_KEY")  # Adaptive Recognition API Key

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Init boto3
s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY,
    endpoint_url=SUPABASE_S3_ENDPOINT,  # Use the S3-specific endpoint here
    region_name=SUPABASE_REGION,
)

# Load YOLO model for motorcycle detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
motorcycle_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov11.pt')
motorcycle_model.to(device)
motorcycle_model.conf = 0.25
motorcycle_model.iou = 0.45
motorcycle_model.classes = [3]  # Motorcycle only

# Init Flask
app = Flask(__name__)

def send_to_adaptive_api(image_path):
    """Send the image to the Adaptive Recognition API and extract the license plate."""
    url = "https://api.cloud.adaptiverecognition.com/vehicle/sas"
    headers = {
        "X-Api-Key": X_API_KEY  # Remove Content-Type header, requests will set it
    }
    
    # Prepare the files and data separately
    files = {
        "image": ("image.jpg", open(image_path, "rb"), "image/jpeg")
    }
    
    data = {
        "location": "IDN",
        "service": "anpr,mmr"
    }

    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        
        # Check if response status is OK
        if response.status_code != 200:
            print(f"⚠️ API returned status code {response.status_code}")
            print(f"Response content: {response.text}")
            return None
            
        response_dict = response.json()  # Use .json() method directly
        print(f"API Response: {response_dict}")

        # Extract the license plate text
        vehicles = response_dict.get("data", {}).get("vehicles", [])
        if vehicles:
            if "plate" in vehicles[0] and vehicles[0]["plate"]["found"]:
                # Try different fields for license plate text
                plate_info = vehicles[0]["plate"]
                plate_text = (plate_info.get("separatedText") or 
                             plate_info.get("text") or 
                             plate_info.get("unicodeText") or "N/A")
                
                print(f"🎯 Found license plate: {plate_text}")
                return plate_text
            else:
                print("📝 Vehicle found but no license plate detected")
        else:
            print("🚫 No vehicles detected in the image")
        
        return None
    except Exception as e:
        print(f"❌ Error calling Adaptive Recognition API: {e}")
        traceback.print_exc()  # Print full traceback for debugging
        return None

# Before sending to API, enhance the image
def enhance_image_for_license_plate(image):
    """Enhance image to improve license plate detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply slight blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Convert back to color for API (if needed)
    enhanced_color = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    
    return enhanced_color

@app.route('/upload', methods=['POST'])
def classify_image():
    print("📥 Menerima request baru di /upload")

    try:
        device_id = None

        # Multipart form image
        if 'image' in request.files:
            image = request.files['image']
            device_id = request.form.get("device_id", "unknown")
            if image.filename == '':
                return jsonify({"error": "No selected file"}), 400

            in_memory_file = BytesIO()
            image.save(in_memory_file)
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # JSON base64 image
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

        # Save the original image temporarily
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, frame)
        
        # Create an enhanced version for license plate detection
        enhanced_frame = enhance_image_for_license_plate(frame)
        enhanced_path = "temp_enhanced.jpg"
        cv2.imwrite(enhanced_path, enhanced_frame)
        
        # Try with enhanced image first, fall back to original if needed
        license_plate_text = send_to_adaptive_api(enhanced_path)
        if not license_plate_text:
            print("🔄 Trying with original image...")
            license_plate_text = send_to_adaptive_api(temp_image_path)
            
        print(f"🔍 License Plate Result: {license_plate_text if license_plate_text else 'No plate detected'}")

        # Step 2: Detect motorcycles using YOLO
        results = motorcycle_model(frame)
        detections = results.xyxy[0].to('cpu')
        detected_motorcycles = [d for d in detections if int(d[5]) == 3]

        if detected_motorcycles:
            highest_conf = max([float(d[4]) for d in detected_motorcycles])
            print(f"🚨 Motor terdeteksi! Confidence: {highest_conf:.2f}")

            # Create a copy of the frame for annotations
            annotated_frame = frame.copy()
            
            # Draw bounding boxes and add license plate text
            for idx, (*xyxy, conf, cls) in enumerate(detected_motorcycles):
                x1, y1, x2, y2 = map(int, xyxy)
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add confidence
                cv2.putText(annotated_frame, f"{conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Extract motorcycle ROI for debugging
                if x1 < x2 and y1 < y2:  # Sanity check for valid coordinates
                    motorcycle_roi = frame[y1:y2, x1:x2]
                    roi_filename = f"{device_id}_roi_{idx}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                    roi_path = os.path.join(saved_dir, roi_filename)
                    cv2.imwrite(roi_path, motorcycle_roi)
            
            # Add license plate text to the frame if available
            if license_plate_text:
                cv2.putText(annotated_frame, f"Plate: {license_plate_text}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Save annotated full frame
            full_filename = f"{device_id}_full_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            full_path = os.path.join(saved_dir, full_filename)
            cv2.imwrite(full_path, annotated_frame)
            
            # Upload to Supabase using your working S3 method
            try:
                with open(full_path, 'rb') as file_data:
                    file_content = file_data.read()
                
                s3.put_object(
                    Bucket=SUPABASE_BUCKET,
                    Key=full_filename,
                    Body=file_content,
                    ContentType="image/jpeg"
                )
                print(f"✅ Image uploaded to Supabase as: {full_filename}")
                
                # Delete temporary files
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                if os.path.exists(enhanced_path):
                    os.remove(enhanced_path)
                
            except Exception as upload_error:
                print(f"❌ Supabase upload error: {upload_error}")
                traceback_str = traceback.format_exc()
                print(f"Stack trace: {traceback_str}")
                
                # Try alternative upload method
                try:
                    print("🔄 Trying alternative upload method...")
                    with open(full_path, 'rb') as file_data:
                        file_content = file_data.read()
                        
                    s3.put_object(
                        Bucket=SUPABASE_BUCKET,
                        Key=full_filename,
                        Body=file_content,
                        ContentType="image/jpeg"
                    )
                    print(f"✅ Image uploaded via S3 client as: {full_filename}")
                except Exception as s3_error:
                    print(f"❌ S3 upload also failed: {s3_error}")
            
            # Return response with all information
            return jsonify({
                "role": "response",
                "result": "motorcycle_and_license_plate" if license_plate_text else "motorcycle_only",
                "motorcycle_detections": len(detected_motorcycles),
                "license_plate_text": license_plate_text if license_plate_text else "N/A",
                "confidence": round(highest_conf, 2),
                "filename": full_filename,
                "image_url": f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{full_filename}"
            }), 200

        else:
            print("✅ No motorcycles detected.")
            
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
                
                # Delete temporary files
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                if os.path.exists(enhanced_path):
                    os.remove(enhanced_path)
                
                return jsonify({
                    "role": "response",
                    "result": "license_plate_only",
                    "motorcycle_detections": 0,
                    "license_plate_text": license_plate_text,
                    "filename": plate_filename
                }), 200
            else:
                # Delete temporary files
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                if os.path.exists(enhanced_path):
                    os.remove(enhanced_path)
                    
                return jsonify({
                    "role": "response",
                    "result": "none",
                    "motorcycle_detections": 0,
                    "license_plate_text": "N/A"
                }), 200

    except Exception as e:
        print("❌ Error:", e)
        traceback_str = traceback.format_exc()
        print(f"Stack trace: {traceback_str}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)