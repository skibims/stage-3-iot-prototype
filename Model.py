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

# Load environment
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_REGION = os.getenv("SUPABASE_REGION")
ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

# Init boto3
s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY,
    endpoint_url=SUPABASE_URL,
    region_name=SUPABASE_REGION,
)

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov11.pt')
model.to(device)
model.conf = 0.25
model.iou = 0.45
model.classes = [3]  # Motorcycle only

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        data = request.get_json()
        device_id = data.get("device_id")
        image_base64 = data.get("image_base64")

        if not device_id or not image_base64:
            return jsonify({"error": "Missing data"}), 400

        # Decode gambar
        img_data = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Deteksi motor
        results = model(frame)
        detections = results.xyxy[0].to('cpu')
        detected_motorcycles = [d for d in detections if int(d[5]) == 3]

        if detected_motorcycles:
            highest_conf = max([float(d[4]) for d in detected_motorcycles])
            print(f"🚨 Motor terdeteksi! Confidence: {highest_conf:.2f}")

            # Tambah tulisan ke gambar
            cv2.putText(frame, "🚨 Ada motor!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Encode dan upload ke Supabase
            _, jpeg = cv2.imencode('.jpg', frame)
            buffer = BytesIO(jpeg.tobytes())
            filename = f"motor_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"

            s3.upload_fileobj(
                buffer,
                SUPABASE_BUCKET,
                filename,
                ExtraArgs={"ContentType": "image/jpeg"}
            )

            return jsonify({
                "role": "response",
                "device_id": device_id,
                "result": "motorcycle",
                "confidence": round(highest_conf, 2),
                "filename": filename
            }), 200

        else:
            print("✅ Tidak ada motor terdeteksi.")
            return jsonify({
                "role": "response",
                "device_id": device_id,
                "result": "none"
            }), 200

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
