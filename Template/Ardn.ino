#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_MOSI   9
#define OLED_CLK   10
#define OLED_DC    11
#define OLED_CS    12
#define OLED_RESET 13
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, OLED_MOSI, OLED_CLK, OLED_DC, OLED_RESET, OLED_CS);


void setup()
{
  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);
  pinMode(11, OUTPUT);
  pinMode(12, OUTPUT);
  pinMode(13, OUTPUT); 
  Serial.begin(9600);
  if(!display.begin(SSD1306_SWITCHCAPVCC)) {
    Serial.println(F("SSD1306 allocation failed"));
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println("AI STATUS: OFFLINE");
  display.display();
}
void loop() {
    if (Serial.available()) {

        // 1. Baca perintah LED dulu
        String ledcmd = Serial.readStringUntil('\n');
        ledcmd.trim();

        if (ledcmd == "LED_A_ON") digitalWrite(2, HIGH);
        else if (ledcmd == "LED_A_OFF") digitalWrite(2, LOW);
        else if (ledcmd == "LED_B_ON") digitalWrite(3, HIGH);
        else if (ledcmd == "LED_B_OFF") digitalWrite(3, LOW);

        // 2. Baca pesan utama
        String oledMsg = Serial.readStringUntil('\n');
        oledMsg.trim();
        display.clearDisplay();
        display.setCursor(0, 0);
        display.println(oledMsg);
        display.display();
    }
}
