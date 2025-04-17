#include <WiFi.h>
#include <esp_camera.h>
#include <HTTPClient.h>

// WiFi credentials
const char* ssid = "BvgC-ZmF6bGViaW1vNTU1";
const char* password = "gak ada.";

// Backend URL
const char* backendUrl = "http://<YOUR_BACKEND_IP>:5000/upload";  // Ganti dengan IP backend Flask kamu

// Pins
const int pirPin = 13;
const int buzzerPin = 14;
const int ledPin = 12;

// Cooldown motion
bool motionDetected = false;
unsigned long lastMotionTime = 0;
const unsigned long motionCooldown = 10000;

// CAMERA_MODEL_AI_THINKER
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

void setup_wifi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.println(WiFi.localIP());
}

void sendImageToBackend() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed!");
    return;
  }

  HTTPClient http;
  WiFiClient client;

  http.begin(client, backendUrl);

  String boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
  String contentType = "multipart/form-data; boundary=" + boundary;
  http.addHeader("Content-Type", contentType);

  // Build body
  String bodyStart = "--" + boundary + "\r\n"
                     "Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n"
                     "Content-Type: image/jpeg\r\n\r\n";

  String bodyEnd = "\r\n--" + boundary + "--\r\n";

  int contentLength = bodyStart.length() + fb->len + bodyEnd.length();
  http.addHeader("Content-Length", String(contentLength));

  // Start connection manually
  int code = http.sendRequest("POST");
  if (code <= 0) {
    Serial.println("❌ Failed to connect to backend!");
    esp_camera_fb_return(fb);
    http.end();
    return;
  }

  // Stream image
  WiFiClient *stream = http.getStreamPtr();
  stream->print(bodyStart);
  stream->write(fb->buf, fb->len);
  stream->print(bodyEnd);

  // Get response
  int responseCode = http.GET();
  String response = http.getString();
  Serial.printf("📤 Image sent. Response code: %d\n", responseCode);
  Serial.println("🧠 Backend response: " + response);

  esp_camera_fb_return(fb);
  http.end();

  // Optional: Activate buzzer/LED if needed
  if (response.indexOf("Motor detected") >= 0) {
    digitalWrite(buzzerPin, HIGH);
    digitalWrite(ledPin, HIGH);
    delay(5000);
    digitalWrite(buzzerPin, LOW);
    digitalWrite(ledPin, LOW);
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(pirPin, INPUT);
  pinMode(buzzerPin, OUTPUT);
  pinMode(ledPin, OUTPUT);
  digitalWrite(buzzerPin, LOW);
  digitalWrite(ledPin, LOW);

  setup_wifi();

  // Camera config
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_HD;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x", err);
    while (true);
  }
}

void loop() {
  int motion = digitalRead(pirPin);
  unsigned long now = millis();

  if (motion == HIGH && !motionDetected && (now - lastMotionTime > motionCooldown)) {
    motionDetected = true;
    lastMotionTime = now;
    Serial.println("🏃 Motion detected! Capturing and sending image...");
    sendImageToBackend();
  }

  if (motion == LOW && motionDetected && (now - lastMotionTime > motionCooldown)) {
    motionDetected = false;
  }
}
