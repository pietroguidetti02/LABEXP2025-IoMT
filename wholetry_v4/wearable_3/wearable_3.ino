// Combined step counting and BLE advertising for wearable with interrupt-based updates
#include <Wire.h>
#include <SparkFun_BMI270_Arduino_Library.h>
#include <bluefruit.h>

// IMU setup
BMI270 myIMU;
uint8_t i2cAddress = BMI2_I2C_PRIM_ADDR; // 0x68
int steps = 0;
unsigned long lastStepTime = 0;
float threshold = 1.4;

// BLE setup
#define NUM_BYTES_BLE_ADDRESS 6
#define UPDATE_INTERVAL 50  // Kept as fallback but will rarely be used

// Timing variables
unsigned long lastBleUpdate = 0;
bool stepDetected = false; // Flag for step detection

// Add to the top of your Arduino code:
BLEService stepService("00001234-0000-1000-8000-00805f9b34fb");
BLECharacteristic stepCharacteristic("00002345-0000-1000-8000-00805f9b34fb", BLERead, sizeof(uint32_t));


void setup() {
  Serial.begin(115200);
  
  // Initialize IMU
  Wire.begin();
  while(myIMU.beginI2C(i2cAddress) != BMI2_OK) {
    Serial.println("Error: BMI270 not connected");
    delay(1000);
  }
  Serial.println("BMI270 connected!");
  
  // Initialize Bluefruit
  Bluefruit.begin();
  // In setup(), add after Bluefruit.begin():
  // Set up Step Service
  stepService.begin();

  // Set up Step Characteristic
  stepCharacteristic.setProperties(BLERead);
  stepCharacteristic.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  stepCharacteristic.setFixedLen(sizeof(uint32_t));
  stepCharacteristic.begin();
  uint32_t initialSteps = 0;
  stepCharacteristic.write(&initialSteps, sizeof(uint32_t));


  Bluefruit.setName("StepTracker");
  
  // Set up advertising
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addName();
  
  // Start advertising
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(80, 80);
  Bluefruit.Advertising.setFastTimeout(30);
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
      stepDetected = true; // Set flag when step detected
      Serial.print("Steps: ");
      Serial.println(steps);
      lastStepTime = currentTime;
    }
  }
  
  // Update BLE immediately when a step is detected
  if (stepDetected) {
    // Update step characteristic 
    uint32_t currentSteps = (uint32_t)steps;
    stepCharacteristic.write(&currentSteps, sizeof(uint32_t));
    
    // Update advertisement right away
    updateAdvertisingData();
    lastBleUpdate = currentTime;
    
    // Reset the flag
    stepDetected = false;
  }
  // Fallback periodic update (rarely used now)
  else if (currentTime - lastBleUpdate > UPDATE_INTERVAL) {
    updateAdvertisingData();
    lastBleUpdate = currentTime;
  }
  
  delay(10); // Keep this to prevent overwhelming the processor
}

void updateAdvertisingData() {
  // Create manufacturer-specific data with ID embedded
  uint8_t advData[7]; // 2 bytes for manufacturer ID + 5 bytes for data
  
  // First two bytes are manufacturer ID (0xFFFF) in little-endian
  advData[0] = 0xFF;  // Low byte of 0xFFFF
  advData[1] = 0xFF;  // High byte of 0xFFFF
  
  // Next byte is a custom identifier (0x01)
  advData[2] = 0x01;
  
  // Next 4 bytes for steps (uint32_t)
  uint32_t stepsValue = (uint32_t)steps;
  advData[3] = stepsValue & 0xFF;
  advData[4] = (stepsValue >> 8) & 0xFF;
  advData[5] = (stepsValue >> 16) & 0xFF;
  advData[6] = (stepsValue >> 24) & 0xFF;
  
  // Stop advertising temporarily
  Bluefruit.Advertising.stop();
  
  // Clear existing data
  Bluefruit.Advertising.clearData();
  
  // Add flags, power, and name again
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addName();
  
  // Add our manufacturer specific data
  Bluefruit.Advertising.addData(BLE_GAP_AD_TYPE_MANUFACTURER_SPECIFIC_DATA, advData, sizeof(advData));
  
  // Restart advertising
  Bluefruit.Advertising.setInterval(80,80);
  Bluefruit.Advertising.start(0);
  
}