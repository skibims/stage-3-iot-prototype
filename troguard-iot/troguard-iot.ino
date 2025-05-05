#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <esp_camera.h>
#include <ArduinoHttpClient.h>
#include <ArduinoJson.h>

// CAMERA_MODEL_AI_THINKER
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// Pins
const int pirPin = 13;
const int buzzerPin = 12;
const int redLEDPin = 15;
const int yellowLEDPin = 14;
const int ldrPin = 2;      // LDR sensor on GPIO 2 (was greenLEDPin)
const int flashPin = 4;

// LDR threshold - adjust based on your environment and LDR specs
const int ldrDarkThreshold = 1800;  // Below this value is considered dark (needs flash)
                                    // For ESP32 analog read: 0 (brightest) to 4095 (darkest)

// Backend config (default, bisa diubah dari portal)
String backendHost = "192.168.200.83";
int backendPort = 5000;
String uploadPath = "/upload";

// WiFi portal
Preferences preferences;
WebServer server(80);
const char* apSSID = "ESP32-CAM-Setup";

// Clients
WiFiClient wifi;
HttpClient http(wifi, backendHost.c_str(), backendPort);

// Motion detection
bool motionDetected = false;
unsigned long lastMotionTime = 0;
const unsigned long motionCooldown = 3000;  // Increased to 3 seconds

// AP timeout
bool apActive = true;
unsigned long apStartTime = 0;

void initPins() {
  pinMode(pirPin, INPUT);
  pinMode(buzzerPin, OUTPUT);
  pinMode(redLEDPin, OUTPUT);
  pinMode(yellowLEDPin, OUTPUT);
  pinMode(ldrPin, INPUT);     // LDR as input
  pinMode(flashPin, OUTPUT);
  
  digitalWrite(buzzerPin, LOW);
  digitalWrite(redLEDPin, LOW);
  digitalWrite(yellowLEDPin, LOW);
  digitalWrite(flashPin, LOW);
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
  config.frame_size = FRAMESIZE_UXGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("\n❌ Camera init failed: 0x%x", err);
    while (true);
  }
  
  resetCamera();
}

void resetCamera() {
  sensor_t *s = esp_camera_sensor_get();
  s->set_brightness(s, 1);
  s->set_contrast(s, 1);
  s->set_saturation(s, 1);
  s->set_gain_ctrl(s, 1);
  s->set_exposure_ctrl(s, 1);
  s->set_awb_gain(s, 1);
  s->set_framesize(s, FRAMESIZE_UXGA);
}

void startAPServer() {
  IPAddress IP = WiFi.softAPIP();
  Serial.println("📶 AP Mode Started: " + IP.toString());

  server.on("/", HTTP_GET, []() {
    server.send(200, "text/html",
      "<h2>WiFi Configuration</h2>"
      "<form action=\"/save_wifi\" method=\"POST\">"
      "SSID: <input name=\"ssid\"><br>"
      "Password: <input name=\"password\" type=\"password\"><br>"
      "<input type=\"submit\" value=\"Save WiFi\"></form>"
      "<hr><h2>Backend Configuration</h2>"
      "<form action=\"/save_backend\" method=\"POST\">"
      "Host: <input name=\"host\" value='" + backendHost + "'><br>"
      "Port: <input name=\"port\" value='" + String(backendPort) + "' type=\"number\"><br>"
      "<input type=\"submit\" value=\"Save Backend\"></form>");
  });

  server.on("/save_wifi", HTTP_POST, []() {
    String ssid = server.arg("ssid");
    String pass = server.arg("password");

    if (ssid.length() > 0) {
      preferences.begin("wifi", false);
      preferences.putString("ssid", ssid);
      preferences.putString("password", pass);
      preferences.end();
      server.send(200, "text/html", "✅ WiFi Saved! Rebooting...");
      delay(2000);
      ESP.restart();
    } else {
      server.send(400, "text/plain", "❌ Incomplete WiFi data");
    }
  });

  server.on("/save_backend", HTTP_POST, []() {
    String host = server.arg("host");
    int port = server.arg("port").toInt();

    if (host.length() > 0 && port > 0) {
      preferences.begin("wifi", false);
      preferences.putString("host", host);
      preferences.putInt("port", port);
      preferences.end();

      // Reinitialize HttpClient with updated backend configuration
      backendHost = host;
      backendPort = port;
      http = HttpClient(wifi, backendHost.c_str(), backendPort);

      server.send(200, "text/html", "✅ Backend Config Saved! Rebooting...");
      delay(2000);
      ESP.restart();
    } else {
      server.send(400, "text/plain", "❌ Incomplete backend config");
    }
  });

  server.begin();
}

bool connectToStoredWiFi() {
  preferences.begin("wifi", true);
  String ssid = preferences.getString("ssid", "");
  String pass = preferences.getString("password", "");
  backendHost = preferences.getString("host", backendHost);
  backendPort = preferences.getInt("port", backendPort);
  preferences.end();

  if (ssid.length() == 0) return false;

  WiFi.begin(ssid.c_str(), pass.c_str());
  Serial.print("🔌 Connecting to " + ssid);

  for (int i = 0; i < 20; i++) {
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\n✅ Connected! IP: " + WiFi.localIP().toString());
      http = HttpClient(wifi, backendHost.c_str(), backendPort);
      return true;
    }
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n❌ Connection failed.");
  return false;
}

void sendMultipartImage(camera_fb_t* fb) {
  String backendURL = backendHost + ":" + String(backendPort) + uploadPath;
  Serial.println("🔗 Sending image to backend:" + backendURL);
  String boundary = "----ESP32FormBoundary";
  String contentType = "multipart/form-data; boundary=" + boundary;

  String bodyStart = "--" + boundary + "\r\n";
  bodyStart += "Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n";
  bodyStart += "Content-Type: image/jpeg\r\n\r\n";
  String bodyEnd = "\r\n--" + boundary + "--\r\n";

  int totalLength = bodyStart.length() + fb->len + bodyEnd.length();

  http.beginRequest();
  http.post(uploadPath.c_str());
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

  if (statusCode != 200) return false;

  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, response);
  if (err) return false;

  return doc["result"] == "motorcycle";
}

void alertMotorDetected() {
  digitalWrite(buzzerPin, HIGH);
  digitalWrite(redLEDPin, HIGH);
  delay(5000);
  digitalWrite(buzzerPin, LOW);
  digitalWrite(redLEDPin, LOW);
}

// Function to read LDR and determine if flash is needed
bool isFlashNeeded() {
  int lightLevel = analogRead(ldrPin);
  Serial.print("💡 Light level: ");
  Serial.print(lightLevel);
  
  // Higher value means darker environment with LDR
  if (lightLevel > ldrDarkThreshold) {
    Serial.println(" - Dark environment, using flash");
    return true;
  } else {
    Serial.println(" - Sufficient light, no flash needed");
    return false;
  }
}

void captureAndSendImage() {
  delay(200);
  
  // Check light conditions and enable flash only if needed
  bool needFlash = isFlashNeeded();
  
  if (needFlash) {
    digitalWrite(flashPin, HIGH);
  }
  
  // Small delay after turning on flash to let it stabilize
  if (needFlash) delay(100);
  
  Serial.println("📸 Taking a new photo now...");
  camera_fb_t * fb = esp_camera_fb_get();
  
  // Turn off flash immediately after capture
  if (needFlash) {
    digitalWrite(flashPin, LOW);
  }
  
  if (!fb) {
    Serial.println("❌ Camera capture failed!");
    return;
  }
  
  Serial.print("📦 Image captured! Size: ");
  Serial.print(fb->len);
  Serial.println(" bytes");
  
  sendMultipartImage(fb);
  esp_camera_fb_return(fb);

  if (handleBackendResponse()) {
    Serial.println("🚨 Motorcycle detected!");
    alertMotorDetected();
  } else {
    Serial.println("✅ No motorcycle detected.");
  }
  
  // Reset camera settings after each capture to ensure fresh images
  resetCamera();
}

void handleMotionDetection() {
  int motion = digitalRead(pirPin);
  unsigned long now = millis();

  // Debug the PIR sensor state
  static int lastMotionState = -1;
  if (motion != lastMotionState) {
    Serial.print("📡 PIR sensor changed to: ");
    Serial.println(motion == HIGH ? "HIGH" : "LOW");
    lastMotionState = motion;
  }

  // Always capture new image when motion is detected and cooldown has elapsed
  if (motion == HIGH && (now - lastMotionTime > motionCooldown)) {
    lastMotionTime = now;
    Serial.println("🏃 Motion detected! Capturing image...");
    captureAndSendImage();
  }

  // Simple state tracking - not used for capture decision
  if (motion == HIGH && !motionDetected) {
    motionDetected = true;
  } else if (motion == LOW && motionDetected) {
    motionDetected = false;
  }
}

void setup() {
  Serial.begin(115200);
  initPins();

  digitalWrite(yellowLEDPin, HIGH);
  
  initCamera();
  

  WiFi.softAP(apSSID);         // Always start AP at boot
  startAPServer();             // Start portal server
  apStartTime = millis();      // Track AP time start

  // Test the LDR sensor
  int initialLight = analogRead(ldrPin);
  Serial.print("Initial light level reading: ");
  Serial.println(initialLight);
  Serial.println("LDR threshold set to: " + String(ldrDarkThreshold));

  // Load WiFi and Backend configuration from preferences
  if (connectToStoredWiFi()) {
    // WiFi connected, continue with normal operation
    Serial.println("✅ WiFi connected, Backend: " + backendHost + ":" + String(backendPort));
  } else {
    Serial.println("❌ Failed to connect to WiFi");
  }
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    // WiFi status indicator now uses only yellowLEDPin
    digitalWrite(yellowLEDPin, LOW);  // Yellow LED off when connected
    handleMotionDetection();
  } else {
    // Blink yellow LED to indicate no WiFi connection
    digitalWrite(yellowLEDPin, millis() % 1000 < 500 ? HIGH : LOW);
    server.handleClient();
  }

  // Auto-disable AP after 5 mins of successful WiFi connection
  if (apActive && WiFi.status() == WL_CONNECTED && millis() - apStartTime > 5 * 60 * 1000) {
    Serial.println("🛑 Disabling AP mode...");
    server.stop();
    WiFi.softAPdisconnect(true);
    apActive = false;
  }
}