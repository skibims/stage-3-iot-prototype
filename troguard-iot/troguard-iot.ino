#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoHttpClient.h>
#include <ArduinoJson.h>
#include <esp_camera.h>

// WiFi credentials
const char* ssid = "Hotspot - UI";
const char* password = "";

// Backend config
const char* backendHost = "10.5.88.42"; // Ganti dengan IP backend Flask kamu
const int backendPort = 5000;
const char* uploadPath = "/upload";

// Pins
const int pirPin = 13;
const int buzzerPin = 14;
const int ledPin = 12;

// Motion detection state
bool motionDetected = false;
unsigned long lastMotionTime = 0;
const unsigned long motionCooldown = 10000;

// CAMERA_MODEL_AI_THINKER
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// Clients
WiFiClient wifi;
HttpClient http(wifi, backendHost, backendPort);

void connectToWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("🔌 Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi connected: " + WiFi.localIP().toString());
}

void initPins() {
  pinMode(pirPin, INPUT);
  pinMode(buzzerPin, OUTPUT);
  pinMode(ledPin, OUTPUT);
  digitalWrite(buzzerPin, LOW);
  digitalWrite(ledPin, LOW);
}

void initCamera() {
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
    Serial.printf("❌ Camera init failed: 0x%x", err);
    while (true);
  }
}

void alertMotorDetected() {
  digitalWrite(buzzerPin, HIGH);
  digitalWrite(ledPin, HIGH);
  delay(5000);
  digitalWrite(buzzerPin, LOW);
  digitalWrite(ledPin, LOW);
}

void sendMultipartImage(camera_fb_t* fb) {
  String boundary = "----ESP32FormBoundary";
  String contentType = "multipart/form-data; boundary=" + boundary;

  String bodyStart = "--" + boundary + "\r\n";
  bodyStart += "Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n";
  bodyStart += "Content-Type: image/jpeg\r\n\r\n";
  String bodyEnd = "\r\n--" + boundary + "--\r\n";

  int totalLength = bodyStart.length() + fb->len + bodyEnd.length();

  http.beginRequest();
  http.post(uploadPath);
  http.sendHeader("Content-Type", contentType);
  http.sendHeader("Content-Length", totalLength);
  http.beginBody();
  http.print(bodyStart);
  http.write(fb->buf, fb->len);
  http.print(bodyEnd);
  http.endRequest();
}

bool handleBackendResponse() {
  int statusCode = http.responseStatusCode();
  String response = http.responseBody();
  Serial.printf("📤 Sent image | Response [%d]: %s\n", statusCode, response.c_str());

  if (statusCode != 200) {
    Serial.println("❌ Backend error");
    return false;
  }

  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, response);
  if (err) {
    Serial.println("❌ JSON parsing failed");
    return false;
  }

  return doc["result"];
}

void captureAndSendImage() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Camera capture failed!");
    return;
  }

  sendMultipartImage(fb);
  esp_camera_fb_return(fb);

  if (handleBackendResponse()) {
    Serial.println("🚨 Motorcycle detected!");
    alertMotorDetected();
  } else {
    Serial.println("✅ No motorcycle detected.");
  }
}

void handleMotionDetection() {
  int motion = digitalRead(pirPin);
  unsigned long now = millis();

  if (motion == HIGH && !motionDetected && (now - lastMotionTime > motionCooldown)) {
    motionDetected = true;
    lastMotionTime = now;
    Serial.println("🏃 Motion detected! Capturing image...");
    captureAndSendImage();
  }

  if (motion == LOW && motionDetected && (now - lastMotionTime > motionCooldown)) {
    motionDetected = false;
  }
}

void setup() {
  Serial.begin(115200);
  initPins();
  connectToWiFi();
  initCamera();
}

void loop() {
  handleMotionDetection();
}
