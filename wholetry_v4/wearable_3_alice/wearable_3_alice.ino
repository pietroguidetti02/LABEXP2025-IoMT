// Combined step counting and BLE advertising for wearable with interrupt-based updates
#include <Wire.h>
#include <SparkFun_BMI270_Arduino_Library.h>
#include <bluefruit.h>

// IMU setup
BMI270 myIMU;
uint8_t i2cAddress = BMI2_I2C_PRIM_ADDR; // 0x68

// BLE setup
BLEService stepService("00001234-0000-1000-8000-00805f9b34fb");
BLECharacteristic stepCharacteristic("00002345-0000-1000-8000-00805f9b34fb", BLERead | BLENotify, sizeof(uint32_t));

int steps = 0;
unsigned long lastStepTime = 0;
float threshold = 1.4;
unsigned long lastBleUpdate = 0;
#define BLE_UPDATE_INTERVAL 500  // update every second if no new step detected

void setup() {
  Serial.begin(115200);

  // Initialize IMU
  Wire.begin();
  while (myIMU.beginI2C(i2cAddress) != BMI2_OK) {
    Serial.println("Error: BMI270 not connected");
    delay(1000);
  }
  Serial.println("BMI270 connected!");

  // Initialize BLE
  Bluefruit.begin();
  Bluefruit.setName("StepTracker");

  // Set up Step Service and Characteristic
  stepService.begin();
  stepCharacteristic.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  stepCharacteristic.setFixedLen(sizeof(uint32_t));
  stepCharacteristic.begin();

  uint32_t initialSteps = 0;
  stepCharacteristic.write(&initialSteps, sizeof(uint32_t));

  // Advertising setup
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addName();

  // Initial manufacturer-specific data
  updateAdvertisingData(initialSteps);

  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(80, 80);  // ~50ms interval
  Bluefruit.Advertising.start(0);

  Serial.println("BLE advertising started");
}

void loop() {
  // Step counting
  myIMU.getSensorData();
  float ax = myIMU.data.accelX;
  float ay = myIMU.data.accelY;
  float az = myIMU.data.accelZ;
  
  // Calculate total acceleration
  float a_tot = sqrt(ax * ax + ay * ay + az * az);
  
  // Step detection
  unsigned long currentTime = millis();
  if (a_tot > threshold) {
    if (currentTime - lastStepTime > 700) {
      steps++;
      Serial.print("Steps: ");
      Serial.println(steps);
      lastStepTime = currentTime;

      // Update characteristic
      stepCharacteristic.write(&steps, sizeof(uint32_t));
      if (stepCharacteristic.notifyEnabled()) {
        stepCharacteristic.notify(&steps, sizeof(uint32_t));
      }

      // Update advertising data dynamically
      updateAdvertisingData(steps);
    }
  }

  // Periodic BLE update to keep data fresh
  if (millis() - lastBleUpdate > BLE_UPDATE_INTERVAL) {
    updateAdvertisingData(steps);
    lastBleUpdate = millis();
  }

  delay(100);  // Small delay to avoid busy loop
}

void updateAdvertisingData(uint32_t stepsValue) {
  uint8_t advData[7];
  advData[0] = 0xFF;  // Manufacturer ID (low byte)
  advData[1] = 0xFF;  // Manufacturer ID (high byte)
  advData[2] = 0x01;  // Custom identifier
  advData[3] = stepsValue & 0xFF;
  advData[4] = (stepsValue >> 8) & 0xFF;
  advData[5] = (stepsValue >> 16) & 0xFF;
  advData[6] = (stepsValue >> 24) & 0xFF;

  // Clear existing manufacturer-specific data before adding new
  Bluefruit.Advertising.clearData();
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addName();
  Bluefruit.Advertising.addData(BLE_GAP_AD_TYPE_MANUFACTURER_SPECIFIC_DATA, advData, sizeof(advData));
}