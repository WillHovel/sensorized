// Program 1: Data collection
//
// Streams labeled flex-sensor readings over Serial as CSV rows so they can
// be captured on a PC into a dataset file used by
// 02_train_model/train_model.py.
//
// Serial protocol (Serial Monitor, 115200 baud, newline-terminated lines):
//   l<index>   select the current label, e.g. "l0" = fist, "l3" = thumbs_up
//              (indices match GESTURE_LABELS in gesture_config.h - sent on
//              boot and on any unrecognized command)
//   s          start streaming samples tagged with the current label
//   x          stop streaming
//
// While streaming, one CSV line per sample is printed:
//   label,ch0,ch1,ch2,ch3,ch4
//
// Workflow: pick a label ("l0"), hold that hand shape, "s", let it run a few
// seconds while you vary the hand slightly (angle, how tightly you're
// holding it), "x", repeat for every gesture - ideally across a few separate
// sessions so the dataset isn't all from one static hand position. Capture
// the Serial Monitor output to a .csv file (or use a simple serial-to-file
// script) - that file is what you point train_model.py at with --data.

#include <gesture_config.h>

int currentLabel = 0;
bool streaming = false;

int readSensorRaw(int pin) {
    long sum = 0;
    for (int i = 0; i < ADC_OVERSAMPLE; i++) sum += analogRead(pin);
    return sum / ADC_OVERSAMPLE;
}

void printHelp() {
    Serial.println(F("=== Flex Sensor Data Collection ==="));
    for (unsigned i = 0; i < NUM_GESTURE_LABELS; i++) {
        Serial.printf("  l%u = %s\n", i, GESTURE_LABELS[i]);
    }
    Serial.println(F("Commands: l<index> select label | s start | x stop"));
}

void setup() {
    Serial.begin(115200);
    delay(300);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db); // full ~0-3.3V input range
    for (unsigned i = 0; i < NUM_SENSORS; i++) {
        pinMode(FLEX_PINS[i], INPUT);
    }
    printHelp();
}

void handleCommand(const String& cmd) {
    if (cmd.length() == 0) return;
    if (cmd[0] == 'l') {
        int idx = cmd.substring(1).toInt();
        if (idx >= 0 && idx < (int)NUM_GESTURE_LABELS) {
            currentLabel = idx;
            Serial.printf("# label -> %s\n", GESTURE_LABELS[currentLabel]);
        } else {
            Serial.println(F("# invalid label index"));
        }
    } else if (cmd == "s") {
        streaming = true;
        Serial.printf("# streaming label=%s\n", GESTURE_LABELS[currentLabel]);
    } else if (cmd == "x") {
        streaming = false;
        Serial.println(F("# stopped"));
    } else {
        printHelp();
    }
}

void loop() {
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        handleCommand(cmd);
    }

    static unsigned long lastSample = 0;
    unsigned long now = millis();
    if (streaming && now - lastSample >= (1000 / SAMPLE_RATE_HZ)) {
        lastSample = now;
        Serial.print(GESTURE_LABELS[currentLabel]);
        for (unsigned i = 0; i < NUM_SENSORS; i++) {
            Serial.print(',');
            Serial.print(readSensorRaw(FLEX_PINS[i]));
        }
        Serial.println();
    }
}
