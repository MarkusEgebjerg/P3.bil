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

bool motorActive = false;

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
  digitalWrite(R_EN, LOW);
  digitalWrite(L_EN, LOW);

  Serial.println("READY");
  lastSignalTime = millis();
}

void loop() {
  // Enable H-Bridge when active
  if (motorActive) {
    digitalWrite(R_EN, HIGH);
    digitalWrite(L_EN, HIGH);
  } else {
    digitalWrite(R_EN, LOW);
    digitalWrite(L_EN, LOW);
  }

  // Safety timeout check
  if (millis() - lastSignalTime > TIMEOUT_MS) {
    if (motorActive) {  // Only log once when transitioning
      Serial.println("TIMEOUT - Stopping motors");
      stopMotors();
      motorActive = false;
    }
  }

  // READ DATA: Only run if data is waiting
  if (Serial.available() > 0) {
    // Read integers directly
    int raw_angle = Serial.parseInt();
    int speed = Serial.parseInt();

    // FIX: Proper buffer clearing
    // Wait for newline, or timeout after 10ms
    unsigned long start = millis();
    while (Serial.available() > 0 && millis() - start < 10) {
      if (Serial.read() == '\n') {
        break;
      }
    }

    // Reset the safety timer
    lastSignalTime = millis();
    motorActive = true;

    // --- STEERING ---
    // Python sends angle * 10 (e.g., -300 to 300)
    // Map -300/300 to servo 60/120
    if(raw_angle > 0) {raw_angle = raw_angle + 20;}
    int servo_val = map(raw_angle, -235, 235, 28+20, 138-20);

    steering.write(servo_val);


    // --- MOTOR CONTROL ---
    if (speed > 0) {
      // Forward
      analogWrite(LPWM, 0);
      analogWrite(RPWM, constrain(speed, 0, 255));
    }
    else if (speed < 0) {
      // Reverse
      analogWrite(RPWM, 0);
      analogWrite(LPWM, constrain(abs(speed), 0, 255));
    }
    else {
      // Stop
      stopMotors();
    }

    // FIX: Optional - Echo back for debugging (comment out for production)
    // Uncomment next line to see what Arduino receives:
    //Serial.print("OK:"); Serial.print(raw_angle); Serial.print(","); Serial.println(speed);
  }

  // FIX: Small delay to prevent watchdog issues and reduce CPU load
  // This doesn't affect responsiveness since serial is buffered
  delay(1);  // 1ms delay = still ~1000 Hz loop rate
}

void stopMotors() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);
  digitalWrite(RPWM, LOW);
  digitalWrite(LPWM, LOW);
}