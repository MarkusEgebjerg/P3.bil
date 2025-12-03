#!/usr/bin/env python
"""
HW-039 Motor Driver using SINGLE PWM pin
"""

import Jetson.GPIO as GPIO
import time


class MotorDriverHW039:
    """
    HW-039 / BTS7960 Motor Driver with single PWM pin

    Pin Configuration:
    - One PWM pin (33) controls speed
    - Two digital pins (32, 35) control direction via RPWM/LPWM
    - Two enable pins (29, 31) enable the driver
    """

    def __init__(
            self,
            pwm_pin=33,  # Hardware PWM pin (speed control)
            dir_pin_r=32,  # Direction control - forward
            dir_pin_l=35,  # Direction control - reverse
            R_EN=29,  # Right enable
            L_EN=31,  # Left enable
            pwm_freq=100,
            board_mode=GPIO.BOARD
    ):
        print(f"Jetson Model: {GPIO.model}")
        GPIO.setwarnings(False)

        self.pwm_pin = pwm_pin
        self.dir_pin_r = dir_pin_r
        self.dir_pin_l = dir_pin_l
        self.R_EN = R_EN
        self.L_EN = L_EN
        self.pwm_freq = pwm_freq

        # Set board mode
        GPIO.setmode(board_mode)
        print(f"GPIO Mode: {GPIO.getmode()}")

        # Setup enable pins (always HIGH when in use)
        GPIO.setup(self.R_EN, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.L_EN, GPIO.OUT, initial=GPIO.HIGH)
        print(f"Enable pins {R_EN}, {L_EN} set to HIGH")

        # Setup direction control pins (digital outputs)
        GPIO.setup(self.dir_pin_r, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.dir_pin_l, GPIO.OUT, initial=GPIO.LOW)
        print(f"Direction pins {dir_pin_r}, {dir_pin_l} initialized")

        # Setup PWM pin (hardware PWM for speed)
        GPIO.setup(self.pwm_pin, GPIO.OUT, initial=GPIO.LOW)
        self.pwm = GPIO.PWM(self.pwm_pin, self.pwm_freq)
        self.pwm.start(0)
        print(f"PWM started on pin {pwm_pin} at {pwm_freq} Hz")

        self.current_speed = 0
        self.direction = "stop"

    # ----------------------------- CONTROL METHODS -----------------------------

    def forward(self, speed=50):
        """
        Drive forward
        Args:
            speed: 0-100 (duty cycle percentage)
        """
        speed = max(0, min(100, speed))

        # Set direction: RPWM=HIGH, LPWM=LOW
        GPIO.output(self.dir_pin_r, GPIO.HIGH)
        GPIO.output(self.dir_pin_l, GPIO.LOW)

        # Set speed via PWM
        self.pwm.ChangeDutyCycle(speed)

        self.direction = "forward"
        self.current_speed = speed
        print(f"Forward: {speed}%")

    def reverse(self, speed=50):
        """
        Drive in reverse
        Args:
            speed: 0-100 (duty cycle percentage)
        """
        speed = max(0, min(100, speed))

        # Set direction: RPWM=LOW, LPWM=HIGH
        GPIO.output(self.dir_pin_r, GPIO.LOW)
        GPIO.output(self.dir_pin_l, GPIO.HIGH)

        # Set speed via PWM
        self.pwm.ChangeDutyCycle(speed)

        self.direction = "reverse"
        self.current_speed = speed
        print(f"Reverse: {speed}%")

    def accelerate(self, target_speed, ramp_time=2.0, direction="forward"):
        """
        Gradually accelerate to target speed

        Args:
            target_speed: Target speed (0-100)
            ramp_time: Time to reach target
            direction: "forward" or "reverse"
        """
        target_speed = max(0, min(100, target_speed))

        # Set direction first
        if direction == "forward":
            GPIO.output(self.dir_pin_r, GPIO.HIGH)
            GPIO.output(self.dir_pin_l, GPIO.LOW)
        else:
            GPIO.output(self.dir_pin_r, GPIO.LOW)
            GPIO.output(self.dir_pin_l, GPIO.HIGH)

        # Ramp up speed
        steps = 50
        step_delay = ramp_time / steps

        print(f"Accelerating {direction} to {target_speed}%")

        for step in range(steps + 1):
            current_speed = (target_speed * step) / steps
            self.pwm.ChangeDutyCycle(current_speed)

            if step % 10 == 0:
                print(f"  Speed: {current_speed:.1f}%")

            time.sleep(step_delay)

        self.direction = direction
        self.current_speed = target_speed
        print(f"Acceleration complete: {target_speed}%")

    def brake(self, brake_time=0.5):
        """
        Gradual deceleration to stop

        Args:
            brake_time: Time to brake in seconds
        """
        if self.current_speed == 0:
            return

        print(f"Braking from {self.current_speed}%")

        steps = 20
        step_delay = brake_time / steps

        for i in range(steps, -1, -1):
            speed = (self.current_speed * i) / steps
            self.pwm.ChangeDutyCycle(speed)
            time.sleep(step_delay)

        self.stop()
        print("Brake complete")

    def stop(self):
        """Immediate stop"""
        # Stop PWM
        self.pwm.ChangeDutyCycle(0)

        # Set both direction pins LOW
        GPIO.output(self.dir_pin_r, GPIO.LOW)
        GPIO.output(self.dir_pin_l, GPIO.LOW)

        self.direction = "stop"
        self.current_speed = 0

    def set_speed(self, speed):
        """
        Change speed without changing direction

        Args:
            speed: 0-100
        """
        if self.direction == "stop":
            print("Warning: Motor is stopped. Use forward() or reverse() first.")
            return

        speed = max(0, min(100, speed))
        self.pwm.ChangeDutyCycle(speed)
        self.current_speed = speed

    # ----------------------------- TEST/DEBUG -----------------------------

    def test_pwm(self, duration=5):
        """Test PWM output by cycling through speeds"""
        print(f"\n{'=' * 60}")
        print(f"Testing PWM on pin {self.pwm_pin}")
        print(f"Direction pins: {self.dir_pin_r} (R), {self.dir_pin_l} (L)")
        print(f"{'=' * 60}\n")

        try:
            # Test forward
            print("Testing FORWARD direction:")
            GPIO.output(self.dir_pin_r, GPIO.HIGH)
            GPIO.output(self.dir_pin_l, GPIO.LOW)

            for duty in range(0, 101, 20):
                self.pwm.ChangeDutyCycle(duty)
                print(f"  Forward: {duty}%")
                time.sleep(duration / 10)

            time.sleep(0.5)

            # Test reverse
            print("\nTesting REVERSE direction:")
            GPIO.output(self.dir_pin_r, GPIO.LOW)
            GPIO.output(self.dir_pin_l, GPIO.HIGH)

            for duty in range(0, 101, 20):
                self.pwm.ChangeDutyCycle(duty)
                print(f"  Reverse: {duty}%")
                time.sleep(duration / 10)

            self.stop()
            print("\nPWM test complete")

        except KeyboardInterrupt:
            print("\nTest interrupted")
            self.stop()

    # ----------------------------- CLEANUP -----------------------------

    def cleanup(self):
        """Cleanup GPIO"""
        print("Cleaning up...")

        try:
            self.stop()
            self.pwm.stop()

            # Disable motor driver
            GPIO.output(self.R_EN, GPIO.LOW)
            GPIO.output(self.L_EN, GPIO.LOW)

            GPIO.cleanup()
            print("Cleanup complete")

        except Exception as e:
            print(f"Cleanup error: {e}")


# ----------------------------- TEST FUNCTION -----------------------------

def test_motor():
    """Complete motor test sequence"""
    print("\n" + "=" * 60)
    print("HW-039 MOTOR DRIVER TEST (Single PWM)")
    print("=" * 60 + "\n")

    motor = MotorDriverHW039(
        pwm_pin=33,  # Hardware PWM
        dir_pin_r=32,  # Forward direction
        dir_pin_l=35,  # Reverse direction
        R_EN=29,
        L_EN=31,
        pwm_freq=100
    )

    try:
        # Test 1: PWM cycling
        print("\n--- Test 1: PWM Cycle Test ---")
        motor.test_pwm(duration=8)
        time.sleep(1)

        # Test 2: Acceleration
        print("\n--- Test 2: Gradual Acceleration ---")
        motor.accelerate(70, ramp_time=3, direction="forward")
        time.sleep(2)

        # Test 3: Speed change
        print("\n--- Test 3: Speed Change ---")
        motor.set_speed(40)
        print("Speed reduced to 40%")
        time.sleep(2)

        # Test 4: Brake
        print("\n--- Test 4: Braking ---")
        motor.brake(brake_time=1.5)
        time.sleep(1)

        # Test 5: Reverse
        print("\n--- Test 5: Reverse ---")
        motor.reverse(60)
        time.sleep(3)

        # Test 6: Stop
        print("\n--- Test 6: Stop ---")
        motor.stop()

        print("\n" + "=" * 60)
        print("All tests complete!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")

    finally:
        motor.cleanup()


if __name__ == "__main__":
    test_motor()