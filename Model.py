import paho.mqtt.client as mqtt
import json
import base64
import cv2
import torch
import numpy as np
from datetime import datetime

# MQTT Configuration
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "skibims/troguard/classify"

# Device ID yang ditarget
EXPECTED_DEVICE_ID = "cam-001"

# Load YOLOv5 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov11.pt')
model.to(device)

# YOLOv5 Configuration
model.conf = 0.25
model.iou = 0.45
model.classes = [3]  # COCO ID untuk motorcycle

# MQTT Callback - Saat terkoneksi
def on_connect(client, userdata, flags, rc):
    print("✅ Connected with result code", rc)
    client.subscribe(MQTT_TOPIC)

# MQTT Callback - Saat menerima pesan
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        device_id = data.get("device_id")
        role = data.get("role")
        image_base64 = data.get("image_base64")

        if role == "request" and device_id == EXPECTED_DEVICE_ID:
            print("📷 Gambar diterima, memproses...")

            # Decode base64 ke frame
            img_data = base64.b64decode(image_base64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Deteksi motor
            results = model(frame)
            detections = results.xyxy[0].to('cpu')
            detected_motorcycles = [d for d in detections if int(d[5]) == 3]

            if detected_motorcycles:
                highest_conf = max([float(d[4]) for d in detected_motorcycles])
                print(f"🚨 Motor terdeteksi! Confidence tertinggi: {highest_conf:.2f}")

                # Kirim balasan via MQTT
                response_payload = {
                    "role": "response",
                    "device_id": device_id,
                    "result": "motorcycle",
                    "confidence": round(highest_conf, 2)
                }
                client.publish(MQTT_TOPIC, json.dumps(response_payload))
                print("📤 Respon terkirim ke ESP32")

                # Tambahkan tulisan dan simpan gambar
                cv2.putText(frame, "🚨 Ada motor!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                filename = f"motor_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Gambar disimpan: {filename}")
            
            else:
                print("✅ Tidak ada motor terdeteksi.")

            # Tampilkan frame
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                client.disconnect()

    except Exception as e:
        print("❌ Error saat memproses pesan:", e)

# Inisialisasi MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connect ke broker
client.connect(MQTT_BROKER, MQTT_PORT, 60)
print(f"📡 Listening on topic: {MQTT_TOPIC} ...")

client.loop_forever()
