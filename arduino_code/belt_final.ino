// #include <AccelStepper.h>
// #include <ezButton.h>

// // === Pin Configuration ===
// #define STEP_PIN 9
// #define DIR_PIN 8
// #define LEFT_SWITCH_PIN 7
// #define RIGHT_SWITCH_PIN 6

// AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);
// ezButton leftSwitch(LEFT_SWITCH_PIN);
// ezButton rightSwitch(RIGHT_SWITCH_PIN);

// // === Variables ===
// long maxPosition = 0;
// bool systemReady = false;

// void setup() {
//   Serial.begin(9600);

//   leftSwitch.setDebounceTime(50);
//   rightSwitch.setDebounceTime(50);

//   stepper.setMaxSpeed(100);
//   stepper.setAcceleration(200);

//   Serial.println("Sent from Arduino: Starting system...");
//   homeSystem();
// }

// void loop() {
//   leftSwitch.loop();
//   rightSwitch.loop();

//   if (!systemReady) return;

//   if (Serial.available() >= 3) {
//     uint8_t pos = Serial.read();
//     uint8_t speedLow = Serial.read();
//     uint8_t speedHigh = Serial.read();
//     int speed = (speedHigh << 8) | speedLow;

//     // Range check
//     if (pos < 10 || pos > 250 || speed < 50 || speed > 1000) {
//       Serial.println("Sent from Arduino: Invalid input. Position: 10–250, Speed: 50–1000.");
//       return;
//     }

//     float ratio = pos / 255.0;
//     long target = (long)(ratio * maxPosition);

//     stepper.setMaxSpeed(speed);
//     stepper.moveTo(target);

//     Serial.print("Sent from Arduino: Moving to ");
//     Serial.print(pos);
//     Serial.print(" (step target: ");
//     Serial.print(target);
//     Serial.println(")");

//     while (stepper.distanceToGo() != 0) {
//       stepper.run();
//       leftSwitch.loop();
//       rightSwitch.loop();
//     }

//     Serial.println("Sent from Arduino: Move complete.");
//   }
// }

// void homeSystem() {
//   Serial.println("Homing system...");

//   // Move left until left switch triggered
//   stepper.moveTo(-100000); // large negative number to ensure movement
//   while (leftSwitch.getState() == HIGH) {
//     leftSwitch.loop();
//     stepper.run();
//   }
//   stepper.stop();
//   stepper.setCurrentPosition(0);
//   Serial.println("Homed to position 0");

//   // Wait until right switch is unpressed (optional for switch bounce)
//   while (rightSwitch.getState() == LOW) {
//     rightSwitch.loop();
//   }

//   // Move right until right switch triggered (find max)
//   Serial.println("Finding maximum position...");
//   stepper.moveTo(100000); // large positive number
//   while (rightSwitch.getState() == HIGH) {
//     rightSwitch.loop();
//     stepper.run();
//   }
//   stepper.stop();
//   maxPosition = stepper.currentPosition();
//   Serial.print("Max position found: ");
//   Serial.println(maxPosition);

//   systemReady = true;  // System ready now

//   // Flush any leftover serial data to avoid junk input after homing
//   while (Serial.available()) {
//     Serial.read();
//   }

//   Serial.println("System ready. Send position (10-250) and speed (50-1000) via serial.");
// }




// edit 1:



#include <AccelStepper.h>
#include <ezButton.h>

// === Pin Configuration ===
#define STEP_PIN 9
#define DIR_PIN 8
#define LEFT_SWITCH_PIN 7
#define RIGHT_SWITCH_PIN 6

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);
ezButton leftSwitch(LEFT_SWITCH_PIN);
ezButton rightSwitch(RIGHT_SWITCH_PIN);

long maxPosition = 0;
bool systemReady = false;

void setup() {
  Serial.begin(9600);

  leftSwitch.setDebounceTime(50);
  rightSwitch.setDebounceTime(50);

  stepper.setMaxSpeed(100);
  stepper.setAcceleration(200);

  homeSystem();
  systemReady = true;

  // Flush any leftover input
  while (Serial.available()) Serial.read();
}

void loop() {
  leftSwitch.loop();
  rightSwitch.loop();

  if (!systemReady) return;

  if (Serial.available() >= 3) {
    uint8_t pos = Serial.read();
    uint8_t speedLow = Serial.read();
    uint8_t speedHigh = Serial.read();
    int speed = (speedHigh << 8) | speedLow;

    // Validate input ranges
    if (pos < 10 || pos > 250 || speed < 50 || speed > 1000) {
      Serial.println("Invalid input: Pos(10-250), Speed(50-1000)");
      return;
    }

    float ratio = pos / 255.0;
    long target = (long)(ratio * maxPosition);

    stepper.setMaxSpeed(speed);
    stepper.moveTo(target);

    Serial.print("Moving to pos: ");
    Serial.print(pos);

    while (stepper.distanceToGo() != 0) {
      stepper.run();
      leftSwitch.loop();
      rightSwitch.loop();
    }

    Serial.println("Move complete.");
  }
}

void homeSystem() {
  // Move left until left switch pressed
  stepper.moveTo(-100000);
  while (leftSwitch.getState() == HIGH) {
    leftSwitch.loop();
    stepper.run();
  }
  stepper.stop();
  stepper.setCurrentPosition(0);

  // Wait for right switch release if pressed
  while (rightSwitch.getState() == LOW) {
    rightSwitch.loop();
  }

  // Move right until right switch pressed
  stepper.moveTo(100000);
  while (rightSwitch.getState() == HIGH) {
    rightSwitch.loop();
    stepper.run();
  }
  stepper.stop();
  maxPosition = stepper.currentPosition();

  Serial.println("System ready");
}
