#pragma once
#include <Arduino.h>

// Thin wrapper so inference.ino doesn't need to know the exact U8g2
// constructor for your screen - see display_driver.cpp if you need to
// change it (e.g. different controller chip, or SPI instead of I2C).

void display_init();
void display_show_gesture(const char* label, float confidence); // confidence in [0,1]
void display_show_message(const char* line1, const char* line2 = nullptr);
