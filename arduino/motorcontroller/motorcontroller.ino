/*
 * AAU Racing - Motor Controller
 *
 * Controls servo steering and H-Bridge motor driver
 * Receives commands via serial: "angle,speed\n"
 *
 * Version: 1.0
 * Board: Arduino Nano/Uno
 */

#include <Servo.h>

Servo steering;
const int SERVO_PIN = 9;

// H-Bridge pins
const int RPWM = 5;  // Right PWM (forward)
const int LPWM = 6;  // Left PWM (reverse)
const int L_EN = 7;  // Left Enable
const int R_EN = 8;  // Right Enable

// Safety timeout
const unsigned long TIMEOUT_MS = 500;  // Stop if no data for 0.5 seconds
unsigned long lastSignalTime = 0;

bool motorActive = false

void setup() {
  Serial.begin(115200);

  // Initialize servo
  steering.attach(SERVO_PIN);
  steering.write(90);  // Center position

  // Initialize motor pins
  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);
  pinMode(L_EN, OUTPUT);
  pinMode(R_EN, OUTPUT);

  // Start with everything off
  digitalWrite(RPWM, LOW);
  digitalWrite(LPWM, LOW);

  Serial.println("READY");
  Serial.println("AAU Racing Motor Controller v1.0");
  lastSignalTime = millis();
}

void loop() {
  if (motorActive == true) {
    // Enable H-Bridge
    digitalWrite(R_EN, HIGH);
    digitalWrite(L_EN, HIGH);
  }



  // Has it been too long since we heard from Python?
  if (millis() - lastSignalTime > TIMEOUT_MS) {
    stopMotors();
    motorActive = false
  }

  // READ DATA: Only run if data is waiting
  if (Serial.available() > 0) {
    // Reset the safety timer
    lastSignalTime = millis();

    // Read integers directly (Faster than String)
    // Expecting: "300,32\n" (Angle*10, Speed)
    int raw_angle = Serial.parseInt();
    int speed = Serial.parseInt();

    // Clear the buffer (read until newline)
    if (Serial.read() == '\n') {

      // --- STEERING ---
      // Python sends angle * 10 (e.g., -300 to 300)
      // Map -300/300 directly to servo 60/120
      int servo_val = map(raw_angle, -300, 300, 60, 120);
      servo_val = constrain(servo_val, 60, 120);  // Safety clamp
      steering.write(servo_val);

      // --- MOTOR CONTROL ---
      // Logic for Forward, Reverse, and Stop
      if (speed > 0) {
        analogWrite(LPWM, 0);
        analogWrite(RPWM, constrain(speed, 0, 255));
      }
      else if (speed < 0) {
        analogWrite(RPWM, 0);
        analogWrite(LPWM, constrain(abs(speed), 0, 255));
      }
      else {
        stopMotors();
      }
    }
  }
}

void stopMotors() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);

  // everything off
  digitalWrite(RPWM, LOW);
  digitalWrite(LPWM, LOW);
}