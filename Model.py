from flask import Flask, request
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import os
import uuid
import boto3
import torch

# Load environment
load_dotenv()

SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
SUPABASE_REGION = os.getenv("SUPABASE_REGION")
SUPABASE_URL = os.getenv("SUPABASE_URL")
ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")

# Init boto3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY,
    endpoint_url=SUPABASE_URL,
    region_name=SUPABASE_REGION,
)

# Load YOLOv5/11 model
model = torch.hub.load('ultralytics/yolov11', 'custom', path='yolov11.pt')
model.conf = 0.25
model.iou = 0.45
model.classes = [3]  # motorcycle

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return {'error': 'Image is required'}, 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')

    # Run detection
    results = model(img)
    detections = results.xyxy[0]
    is_motor = any(int(d[5]) == 3 for d in detections)

    if is_motor:
        filename = f"motor_{uuid.uuid4().hex}.jpg"
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        s3.upload_fileobj(
            buffer,
            SUPABASE_BUCKET,
            filename,
            ExtraArgs={"ContentType": "image/jpeg"}
        )

        return {"message": "Motor detected and saved", "filename": filename}
    else:
        return {"message": "No motorcycle detected"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
