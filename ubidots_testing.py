import requests
import time
import random

UBIDOTS_TOKEN = "BBUS-pD4RlQvBNEH2oGPFexPZCd2LpAOVW5"
DEVICE_LABEL = "esp32-cam"

URL = f"https://industrial.api.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}"
HEADERS = {
    "X-Auth-Token": UBIDOTS_TOKEN,
    "Content-Type": "application/json"
}

def send_dummy_data():
    total_pelanggar = random.randint(1, 5)
    jenis = random.choice(["motor", "mobil"])
    

    payload = {
        "pelanggar": total_pelanggar,
        jenis: 1
    }

    response = requests.post(URL, headers=HEADERS, json=payload)
    print("Sent:", payload, "| Status:", response.status_code)

# Kirim data tiap 10 detik untuk simulasi
while True:
    send_dummy_data()
    time.sleep(10)
