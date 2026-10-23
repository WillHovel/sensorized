// Standalone OLED test - NOT part of the main pipeline, just for checking
// that the screen itself is wired and working before you rely on it inside
// 03_inference. Uses GestureShared's pin config and GestureDisplay's display
// wrapper directly, with no sensors or model involved.
//
// Requires: U8g2 library installed, and GestureShared + GestureDisplay
// copied into your Arduino libraries folder (same as the main sketches).

#include <gesture_config.h>
#include <display_driver.h>

void setup() {
    display_init();
    display_show_message("OLED test", "If you see this,");
}

void loop() {
    static int count = 0;
    delay(1000);
    char line[24];
    snprintf(line, sizeof(line), "tick %d", count++);
    display_show_message("OLED test", line);
}
