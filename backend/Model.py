from flask import Flask, request
from supabase import create_client
from dotenv import load_dotenv
import os
from io import BytesIO
from PIL import Image
import uuid
import torch

# Load ENV
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load model
model = torch.hub.load('ultralytics/yolov11', 'custom', path='yolov11.pt')
model.conf = 0.25
model.iou = 0.45
model.classes = [3]  # motorcycle only

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return {"error": "No image uploaded"}, 400

    img_file = request.files['image']
    img = Image.open(img_file.stream).convert("RGB")
    results = model(img)
    detections = results.xyxy[0]

    is_motor = any(int(d[5]) == 3 for d in detections)
    if is_motor:
        filename = f"motor_{uuid.uuid4().hex}.jpg"
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        # Upload ke Supabase Storage
        supabase.storage.from_("motor-images").upload(filename, buffer, {"content-type": "image/jpeg"})
        
        return {"message": "Motor detected and saved", "filename": filename}
    else:
        return {"message": "No motorcycle detected"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
