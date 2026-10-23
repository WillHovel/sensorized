// Program 6: On-device benchmark for validation measurements E, F, G.
//
// Flash this INSTEAD of 03_inference, then run  05_validation/bench_capture.py.
// It reuses your real GestureModel / GestureDisplay libraries, so the numbers
// describe the code you actually ship.
//
// Serial protocol (115200 baud, one command per line, every reply ends "#done"):
//   i              info: CPU MHz, sketch flash bytes, heap, parameter/MAC counts
//   e<ch>,<ticks>  E: ADC noise. For <ticks> ticks at SAMPLE_RATE_HZ, print
//                  E,<ch>,r0..r7  = 8 back-to-back raw reads of FLEX_PINS[ch]
//   f              F: inference time, 5 x 1000 forward passes -> F,min,max,mean (us)
//   g              G: firmware latency, 200 cycles -> G,adc_us,infer_us,oled_us
//                  (ADC read of all channels, normalize+forward, OLED update)

#include <gesture_config.h>
#include <nn_model.h>
#include <display_driver.h>

volatile float sink; // stops the compiler optimising the benchmark away

static void waitTick(unsigned long& next) {
    while ((long)(millis() - next) < 0) {}
    next += 1000 / SAMPLE_RATE_HZ;
}

static void cmdInfo() {
    long params = 0, macs = 0;
    int in = NN_INPUT_SIZE;
    for (int l = 0; l < NN_NUM_LAYERS; l++) {
        params += (long)in * nn_layer_sizes[l] + nn_layer_sizes[l];
        macs += (long)in * nn_layer_sizes[l];
        in = nn_layer_sizes[l];
    }
    Serial.printf("I,cpu_mhz,%u\n", getCpuFrequencyMhz());
    Serial.printf("I,sketch_bytes,%u\n", ESP.getSketchSize());
    Serial.printf("I,heap_total,%u\n", ESP.getHeapSize());
    Serial.printf("I,heap_free,%u\n", ESP.getFreeHeap());
    Serial.printf("I,heap_min_free,%u\n", ESP.getMinFreeHeap());
    Serial.printf("I,params,%ld\n", params);
    Serial.printf("I,macs,%ld\n", macs);
    Serial.printf("I,classes,%d\n", NUM_GESTURES);
    Serial.printf("I,oversample,%d\n", ADC_OVERSAMPLE);
}

static void cmdE(int ch, int ticks) {
    if (ch < 0 || ch >= (int)NUM_SENSORS) { Serial.println("# bad channel"); return; }
    unsigned long next = millis();
    for (int t = 0; t < ticks; t++) {
        waitTick(next);
        int r[8];
        for (int k = 0; k < 8; k++) r[k] = analogRead(FLEX_PINS[ch]);
        Serial.printf("E,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", ch, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
    }
}

static void readAll(float* raw) {
    for (unsigned i = 0; i < NUM_SENSORS; i++) {
        long s = 0;
        for (int k = 0; k < ADC_OVERSAMPLE; k++) s += analogRead(FLEX_PINS[i]);
        raw[i] = (float)(s / ADC_OVERSAMPLE);
    }
}

static void cmdF() {
    float raw[NUM_SENSORS], norm[NN_INPUT_SIZE], probs[NUM_GESTURES];
    readAll(raw);
    float lo = 1e9f, hi = 0, sum = 0;
    for (int rep = 0; rep < 5; rep++) {
        unsigned long t0 = micros();
        for (int i = 0; i < 1000; i++) {
            raw[0] += (i & 1) ? 1.0f : -1.0f; // keep the input changing
            nn_normalize(raw, norm);
            nn_forward(norm, probs);
            sink = probs[0];
        }
        float us = (micros() - t0) / 1000.0f; // per inference incl. normalize
        if (us < lo) lo = us;
        if (us > hi) hi = us;
        sum += us;
    }
    Serial.printf("F,%.2f,%.2f,%.2f\n", lo, hi, sum / 5);
}

static void cmdG() {
    float raw[NUM_SENSORS], norm[NN_INPUT_SIZE], probs[NUM_GESTURES];
    unsigned long next = millis();
    for (int n = 0; n < 200; n++) {
        waitTick(next);
        unsigned long t0 = micros();
        readAll(raw);
        unsigned long t1 = micros();
        nn_normalize(raw, norm);
        nn_forward(norm, probs);
        int best = nn_argmax(probs, NUM_GESTURES);
        unsigned long t2 = micros();
        display_show_gesture(class_labels[best], probs[best]);
        unsigned long t3 = micros();
        Serial.printf("G,%lu,%lu,%lu\n", t1 - t0, t2 - t1, t3 - t2);
    }
}

void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db); // same as 01/03
    for (unsigned i = 0; i < NUM_SENSORS; i++) pinMode(FLEX_PINS[i], INPUT);
    display_init();
    display_show_message("Bench mode", "run bench_capture.py");
    Serial.println("# bench ready");
}

void loop() {
    if (!Serial.available()) return;
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) return;
    if (cmd[0] == 'i') cmdInfo();
    else if (cmd[0] == 'f') cmdF();
    else if (cmd[0] == 'g') cmdG();
    else if (cmd[0] == 'e') {
        int comma = cmd.indexOf(',');
        int ch = cmd.substring(1, comma < 0 ? cmd.length() : comma).toInt();
        int ticks = comma < 0 ? 1500 : cmd.substring(comma + 1).toInt();
        cmdE(ch, ticks);
    }
    Serial.println("#done");
}
