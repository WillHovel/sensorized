#include "display_driver.h"
#include "gesture_config.h"
#include <U8g2lib.h>
#include <Wire.h>

// -----------------------------------------------------------------------
// Confirmed hardware: a 4-pin (VCC/GND/SCL/SDA) 0.96" 128x64 monochrome
// OLED module labeled "GME12864" - this is a very common generic module
// built around the SSD1306 driver, on I2C. If yours turns out to behave
// differently, swap the constructor line below for the matching one from
// the U8g2 constructor list: https://github.com/olikraus/u8g2/wiki/u8g2setupcpp
// Everything else in this file (and in inference.ino) stays the same either
// way, since they only call the display_* functions declared in the header.
// -----------------------------------------------------------------------
static U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE);

void display_init() {
    Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN);
    u8g2.begin();
    u8g2.setFont(u8g2_font_ncenB08_tr);
}

void display_show_gesture(const char* label, float confidence) {
    char pct[8];
    snprintf(pct, sizeof(pct), "%d%%", (int)(confidence * 100.0f + 0.5f));

    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_ncenB14_tr);
    int w = u8g2.getStrWidth(label);
    u8g2.drawStr((128 - w) / 2, 30, label);
    u8g2.setFont(u8g2_font_ncenB08_tr);
    w = u8g2.getStrWidth(pct);
    u8g2.drawStr((128 - w) / 2, 55, pct);
    u8g2.sendBuffer();
}

void display_show_message(const char* line1, const char* line2) {
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_ncenB08_tr);
    u8g2.drawStr(0, 20, line1);
    if (line2) u8g2.drawStr(0, 40, line2);
    u8g2.sendBuffer();
}
