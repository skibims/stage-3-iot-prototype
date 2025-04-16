import torch
import cv2
import time
import os

# Ganti dengan IP ESP32-CAM kamu
ESP32_STREAM_URL = 'http://192.168.185.79:81/stream'

# Load model YOLOv11 (gunakan yolov5 API sebagai loader)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov11.pt')
model.to(device)

# Konfigurasi
model.conf = 0.25
model.iou = 0.45
model.classes = [3]  # COCO class ID untuk motorcycle

# Buat folder untuk simpan gambar
os.makedirs("hasil_motor", exist_ok=True)

# Waktu terakhir menyimpan gambar
last_capture_time = 0
capture_interval = 5  # detik

# Buka stream dari ESP32-CAM
cap = cv2.VideoCapture(ESP32_STREAM_URL)

if not cap.isOpened():
    print("❌ Gagal membuka stream ESP32-CAM.")
else:
    print("✅ Stream berhasil dibuka. Tekan 'q' untuk keluar.")

    motorcycle_total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Gagal membaca frame.")
            break

        results = model(frame)
        detections = results.xyxy[0].to('cpu')
        detected_motorcycles = [d for d in detections if int(d[5]) == 3]
        motorcycle_total += len(detected_motorcycles)

        # Kalau ada motor
        if detected_motorcycles:
            cv2.putText(frame, "🚨 Ada motor!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Simpan gambar jika cukup waktu berlalu
            current_time = time.time()
            if current_time - last_capture_time > capture_interval:
                filename = f"hasil_motor/motor_{int(current_time)}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Motor terdeteksi, gambar disimpan: {filename}")
                last_capture_time = current_time

        # Gambar bounding box dan label
        for *xyxy, conf, cls in detected_motorcycles:
            label = f"motorcycle {conf:.2f}"
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Tampilkan frame
        cv2.imshow('ESP32-CAM Motorcycle Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total motorcycles detected: {motorcycle_total}")
