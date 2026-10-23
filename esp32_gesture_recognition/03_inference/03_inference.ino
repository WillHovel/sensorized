// Program 3: Real-time gesture inference
//
// Reads the 5 flex sensors, runs the trained network (GestureModel's
// nn_model), and shows the recognized gesture + confidence on the OLED
// (GestureDisplay's display_driver). Reports every inference over BOTH USB
// serial AND Bluetooth LE at the same time, so you can use whichever fits -
// USB on the bench, BLE untethered in the field. Nothing else about the
// sketch changes based on which one you're using.
//
// Before building this sketch:
//   1. Run 02_train_model/train_model.py on your collected data.
//   2. Copy its output, model_weights.h, into
//      Documents/Arduino/libraries/GestureModel/src/ (next to nn_model.h).
//
// ---------------------------------------------------------------------
// USB serial protocol (115200 baud) - also what 04_gui/gesture_gui.py uses
// over USB, but works fine typed by hand in the Serial Monitor too:
//   r<hz>   change the polling/inference rate at runtime, e.g. "r20" for
//           20 Hz. No reflash needed. Clamped to 1-200 Hz.
// Every inference is printed as a CSV line:
//   gesture,confidence
//
// ---------------------------------------------------------------------
// BLE: the board advertises itself as "GestureRecognizer" with one custom
// service containing two characteristics:
//   - DATA (read/notify) - same "gesture,confidence" text as the serial
//     line, pushed as a notification after every inference. Any generic
//     BLE scanner app (e.g. nRF Connect on Android/iOS, LightBlue on iOS)
//     can subscribe to this and see live text with no custom app needed.
//   - CMD (write) - same "r<hz>" polling-rate command as serial, so you can
//     retune it over BLE too with nothing plugged in.
// 04_gui/gesture_gui.py's BLE mode talks to exactly this.

#include <gesture_config.h>
#include <nn_model.h>
#include <display_driver.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define ENABLE_BLE 1 // set to 0 to build without BLE (USB serial still works either way)

#define BLE_DEVICE_NAME "GestureRecognizer"
#define SERVICE_UUID    "6f23cf20-1efb-4a2e-9d35-9d1b6a8d8d01"
#define DATA_CHAR_UUID  "6f23cf20-1efb-4a2e-9d35-9d1b6a8d8d02"
#define CMD_CHAR_UUID   "6f23cf20-1efb-4a2e-9d35-9d1b6a8d8d03"

#define VOTE_WINDOW 5      // majority-vote smoothing over this many inferences
#define MIN_CONFIDENCE 0.6f
#define MIN_POLL_HZ 1
#define MAX_POLL_HZ 200

int pollHz = SAMPLE_RATE_HZ; // runtime-adjustable, starts at the config default

int voteHistory[VOTE_WINDOW];
int voteIndex = 0;
int voteCount = 0;

#if ENABLE_BLE
BLEServer* bleServer = nullptr;
BLECharacteristic* bleDataChar = nullptr;
bool bleConnected = false;
#endif

void handleCommand(const String& cmd) {
    if (cmd.length() == 0) return;
    if (cmd[0] == 'r') {
        int hz = cmd.substring(1).toInt();
        if (hz < MIN_POLL_HZ) hz = MIN_POLL_HZ;
        if (hz > MAX_POLL_HZ) hz = MAX_POLL_HZ;
        pollHz = hz;
        Serial.printf("# poll_hz -> %d\n", pollHz);
    }
}

#if ENABLE_BLE
class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* s) override {
        bleConnected = true;
        Serial.println("# BLE client connected");
    }
    void onDisconnect(BLEServer* s) override {
        bleConnected = false;
        Serial.println("# BLE client disconnected");
        delay(200);           // let the stack settle before re-advertising
        s->getAdvertising()->start();
    }
};

class CmdCharCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* c) override {
        String cmd = String(c->getValue().c_str());
        cmd.trim();
        handleCommand(cmd);
    }
};

void bleInit() {
    BLEDevice::init(BLE_DEVICE_NAME);
    bleServer = BLEDevice::createServer();
    bleServer->setCallbacks(new ServerCallbacks());

    BLEService* service = bleServer->createService(SERVICE_UUID);

    bleDataChar = service->createCharacteristic(
        DATA_CHAR_UUID, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    bleDataChar->addDescriptor(new BLE2902()); // required for notify to work
    bleDataChar->setValue("waiting...");

    BLECharacteristic* cmdChar = service->createCharacteristic(
        CMD_CHAR_UUID, BLECharacteristic::PROPERTY_WRITE);
    cmdChar->setCallbacks(new CmdCharCallbacks());

    service->start();
    bleServer->getAdvertising()->addServiceUUID(SERVICE_UUID);
    bleServer->getAdvertising()->start();
}
#endif

int readSensorRaw(int pin) {
    long sum = 0;
    for (int i = 0; i < ADC_OVERSAMPLE; i++) sum += analogRead(pin);
    return sum / ADC_OVERSAMPLE;
}

int majorityVote() {
    int counts[NUM_GESTURES] = {0};
    for (int i = 0; i < voteCount; i++) counts[voteHistory[i]]++;
    int best = 0;
    for (int i = 1; i < NUM_GESTURES; i++)
        if (counts[i] > counts[best]) best = i;
    return best;
}

void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
    for (unsigned i = 0; i < NUM_SENSORS; i++) pinMode(FLEX_PINS[i], INPUT);

    display_init();
    display_show_message("Gesture recognizer", "warming up...");

#if ENABLE_BLE
    bleInit();
    display_show_message("Gesture recognizer", "BLE advertising...");
#endif
    delay(500);
}

void loop() {
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        handleCommand(cmd);
    }

    static unsigned long lastSample = 0;
    unsigned long now = millis();
    if (now - lastSample < (unsigned long)(1000 / pollHz)) return;
    lastSample = now;

    float raw[NUM_SENSORS];
    for (unsigned i = 0; i < NUM_SENSORS; i++) raw[i] = (float)readSensorRaw(FLEX_PINS[i]);

    float normalized[NN_INPUT_SIZE];
    nn_normalize(raw, normalized);

    float probs[NUM_GESTURES];
    nn_forward(normalized, probs);

    int predicted = nn_argmax(probs, NUM_GESTURES);
    voteHistory[voteIndex] = predicted;
    voteIndex = (voteIndex + 1) % VOTE_WINDOW;
    if (voteCount < VOTE_WINDOW) voteCount++;

    int smoothed = majorityVote();
    float confidence = probs[predicted];

    if (confidence >= MIN_CONFIDENCE) {
        display_show_gesture(class_labels[smoothed], confidence);
    } else {
        display_show_message("...", "");
    }

    char line[40];
    snprintf(line, sizeof(line), "%s,%.3f", class_labels[predicted], confidence);
    Serial.println(line);

#if ENABLE_BLE
    if (bleConnected) {
        bleDataChar->setValue((uint8_t*)line, strlen(line));
        bleDataChar->notify();
    }
#endif
}
