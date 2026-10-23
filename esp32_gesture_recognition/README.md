# ESP32-S3 Flex Sensor Gesture Recognizer

Three programs, meant to be run in order, plus an optional desktop GUI that
wraps all of them:

1. **01_data_collection** (Arduino, on-device) - streams labeled flex sensor
   readings over Serial so you can build a training dataset.
2. **02_train_model** (Python, on your PC) - trains a small feedforward net
   (Dense+ReLU hidden layers -> Dense+softmax output) on that dataset and
   exports it as a C header.
3. **03_inference** (Arduino, on-device) - loads that header and classifies
   gestures live, driving the OLED. Reports every inference over **both**
   USB serial and Bluetooth LE at the same time (advertises as
   `GestureRecognizer`), so you can use whichever fits - tethered on the
   bench, untethered in the field. Polling rate can be changed at runtime
   over either one (no reflash) - send `r<hz>` (e.g. `r20`).
4. **04_gui/gesture_gui.py** (Python, on your PC) - a Tkinter control panel
   covering the whole workflow: edit the gesture list, collect data, train
   and keep named/reloadable trainings, deploy one, then run the model live
   with an adjustable polling rate and export a session log as CSV. See
   "Using the GUI" below.

Shared code lives in three small Arduino libraries, split up because Arduino
compiles every .cpp file in a library as soon as any part of it is included
- if unrelated code shared one library, a sketch could fail to build just
for including an unrelated header:
- `GestureShared/` - pin/label config (`gesture_config.h`). Header-only, used
  by all sketches, never drags in anything else.
- `GestureDisplay/` - the OLED wrapper. Used by `03_inference` and the
  standalone `oled_test` sketch. No dependency on a trained model.
- `GestureModel/` - the forward-pass math. Only `03_inference` uses this
  one, since it's the only sketch that needs a trained model to exist.

## Display

Confirmed from a photo: a 4-pin (VCC/GND/SCL/SDA) 0.96" 128x64 monochrome
OLED module labeled "GME12864" - a common generic module built around the
SSD1306 driver, on I2C. `GestureShared/src/display_driver.cpp` is set up for
exactly this. If yours behaves differently, swap one line in that file for
the matching U8g2 constructor (comment in the file links directly to the
list) - nothing else needs to change.

## Setup

- **Board**: select "ESP32S3 Dev Module" in the Arduino IDE.
- **Library**: install "U8g2" (by oliver) via Library Manager.
- **Shared libraries**: copy `GestureShared/`, `GestureDisplay/`, and
  `GestureModel/` into your Arduino libraries directory (e.g.
  `Documents/Arduino/libraries/GestureShared`, `.../GestureDisplay`,
  `.../GestureModel`) so the sketches can `#include <gesture_config.h>`,
  `<display_driver.h>`, `<nn_model.h>`, etc.
- **BLE**: no extra library install needed - it's built into the ESP32
  Arduino core (`BLEDevice.h` etc.), already `#include`d in `03_inference`.
- **Python**: `pip install numpy` (that's the only dependency for training;
  the GUI has its own `requirements.txt`, see "Using the GUI" below).

## Wiring

- 5 flex sensors, each as a voltage divider (3V3 -> flex sensor -> ADC pin ->
  fixed resistor -> GND). Per the sensor's own datasheet, use a **10.0kΩ**
  fixed resistor - the sensor itself sits around 10kΩ at rest, so this
  centers the ADC reading mid-range when the sensor is flat.
- Flex sensors go on **GPIO1-GPIO5** (`FLEX_PINS` in `gesture_config.h`) -
  these are ADC1 channels, unaffected by Wi-Fi/BLE, and not strapping pins.
- OLED on I2C: **SDA = GPIO8, SCL = GPIO9** by default (`OLED_SDA_PIN` /
  `OLED_SCL_PIN` in `gesture_config.h`) - check your specific dev board's
  pinout and adjust if needed.
- MPU6050 (when you add it): shares the I2C bus with the display, default
  address `0x68`.

## Testing the display on its own

`oled_test/oled_test.ino` is a tiny standalone sketch that just shows a
counter on the screen - no sensors, no model. Flash it first if you want to
confirm the OLED and its wiring work before dealing with the rest of the
pipeline. It only needs `GestureShared` and `GestureDisplay` installed (not
`GestureModel`).

## Using the GUI

`04_gui/gesture_gui.py` covers everything below, if you'd rather not do it
by hand:

```
cd 04_gui
pip install -r requirements.txt
python gesture_gui.py
```

`requirements.txt` installs both pyserial and bleak. If you only ever use
USB, bleak isn't required - the app still runs, the BLE radio button just
tells you to install it if you pick it.

At the top of the window, pick **USB** or **BLE**:
- USB: pick a port and Connect, same as always.
- BLE: enter the device name (`GestureRecognizer` by default - matches what
  `03_inference.ino` advertises) and Connect. It scans for and connects to
  that name automatically; no pairing step needed.

Only `03_inference` has BLE - `01_data_collection` is USB only, so use USB
mode for the Collect tab.

- **Gestures tab** - browse to your `gesture_config.h`
  (`.../libraries/GestureShared/src/gesture_config.h`), add/remove gestures,
  save. Re-upload `01_data_collection` afterward to apply it on-device.
- **Collect tab** - connect to the board (running `01_data_collection`),
  pick a label, Start/Stop streaming - rows get appended straight to a CSV
  you choose, with live per-gesture counts.
- **Train tab** - the tuning steps are printed right in the tab. Point it at
  your dataset, click Train, and watch accuracy live. Save a training under
  a name to keep it around (nothing is overwritten), and Deploy copies the
  chosen one into your `GestureModel` library - then re-upload
  `03_inference`.
- **Run tab** - connect to the board (running `03_inference`), adjust the
  polling rate live (no reflash needed), optionally apply a named "gesture
  meaning" mapping (e.g. remap `fist` -> `Resting` for an ethogram, or
  `thumbs_up` -> `Healthy Native Species` for a transect tally - useful for
  the field-logging use cases below), then Start logging and Export to a
  CSV you can open directly in Excel or import into Google Sheets.

## Field use

The Run tab's mapping profiles are meant for exactly this kind of thing:
train one gesture set, then reuse it across different field studies by
swapping which mapping profile is active, no retraining needed.

- **Behavioral ethograms** - map gestures to behavioral states (e.g.
  `fist` -> `Resting`, `triton` -> `Traveling`), keep hands on the
  binoculars, export the timestamped log afterward.
- **Transect tallying** - map gestures to categories (e.g. `thumbs_up` ->
  `Healthy native species`, `middle_finger` -> `Invasive seedling`), tally
  hands-free while the other hand holds a quadrat or sample.
- **Water quality grading** - map gestures to grade levels (e.g. `fist` ->
  `Grade 1`, ... `middle_finger` -> `Grade 5`), signal a colorimetric
  reading without touching a device with contaminated gloves.

In all three, the ESP32 only ever reports the raw gesture it was trained
on (`fist`, `triton`, etc.) - the mapping to a study's actual categories
happens entirely in the GUI, so the same trained model serves any of them.

## Going untethered (BLE)

`03_inference` advertises as **GestureRecognizer** over BLE at the same
time it's available over USB - nothing needs reflashing to switch between
tethered bench testing and untethered field use.

**From a laptop/Raspberry Pi:** use `04_gui/gesture_gui.py`'s BLE mode
(above) - same Run tab, same mapping profiles, same CSV export, just over
BLE instead of USB.

**From a phone, with no custom app:** use a generic BLE scanner app -
[nRF Connect](https://www.nordicsemi.com/Products/Development-tools/nrf-connect-for-mobile)
(Android/iOS) or LightBlue (iOS) both work. Scan, connect to
"GestureRecognizer", open its one custom service, and enable notifications
on the first characteristic (readable/notify) - you'll see live
`gesture,confidence` text appear with every gesture. The second
characteristic (write-only) accepts the same `r<hz>` polling-rate command
if you want to adjust that from the phone too. There's no logging/export
in this path, just live viewing - for field data collection, the GUI over
BLE is the one with CSV export.

**Range:** realistically 10-30m indoors, 50-100m line-of-sight outdoors
with the standard BLE PHY - BLE 5's long-range mode can stretch further but
isn't enabled here since it's unnecessary (the receiving device in all
three scenarios above is on the same person). Body tissue and open water
both attenuate 2.4GHz meaningfully, so expect somewhat less than those
numbers in the marine case specifically - worth testing in your actual
conditions before relying on it.

## Manual workflow

If you'd rather not use the GUI, here's the same process by hand:

1. Flash **01_data_collection**. Open the Serial Monitor at 115200 baud.
   Select a label (`l0` = fist, `l1` = knife_hand, ... printed on boot),
   hold that hand shape, send `s` to start streaming, let it run a few
   seconds while you vary the hand slightly, send `x` to stop. Repeat for
   every gesture, ideally across more than one sitting so the data isn't all
   from one static position. Capture the streamed CSV lines
   (`label,ch0,ch1,ch2,ch3,ch4`) into a file, e.g. `dataset.csv`.
2. Run the trainer:
   ```
   cd 02_train_model
   python train_model.py --data dataset.csv
   ```
   This prints train/validation accuracy as it trains and writes
   `model_weights.h`. Useful flags:
   - `--hidden 24 16` - change hidden layer sizes (default `16 12`)
   - `--epochs 500` - train longer
   - `--data "data/*.csv"` - combine multiple recording sessions
3. Flash **03_inference** after copying the generated `model_weights.h` into
   `Documents/Arduino/libraries/GestureModel/src/` (overwriting the previous
   one - it must sit next to `nn_model.h`, not in the sketch folder). It
   reads the sensors, classifies, smooths the prediction with a small
   majority vote, and shows the gesture + confidence on the OLED. It also
   prints predictions over Serial for debugging.

## Adding a new gesture

1. Add its name to `GESTURE_LABELS` in `GestureShared/src/gesture_config.h`
   (this only controls the on-device labeling menu in step 1 above).
2. Collect data for it the same way as the others.
3. Re-run `train_model.py`. The output layer size, `class_labels[]`, and
   everything else in the generated header update automatically to match
   however many distinct labels are present in your data - no code changes
   needed in `nn_model.h/.cpp` or `inference.ino`.

## Adding the MPU6050 later

- Add IMU reads to both sketches alongside the flex sensor reads, and
  append those columns to the CSV rows in `01_data_collection`.
- `NN_INPUT_SIZE` will grow accordingly the next time you run
  `train_model.py` (it's just `X.shape[1]` from your CSV) - no manual edits
  to the header needed, just re-collect data with the extra columns and
  retrain.
