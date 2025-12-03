import Jetson.GPIO as GPIO
import time


class MotorDriverHW039:
    """
    HW-039 / BTS7960 High-Power Motor Driver Controller
    using Jetson.GPIO

    FIXED VERSION - addresses PWM output issues
    """

    def __init__(
            self,
            rpwm_pin=32,  # PWM pin for forward
            lpwm_pin=33,  # PWM pin for reverse
            R_EN=29,  # Right enable (digital output)
            L_EN=31,  # Left enable (digital output)
            pwm_freq=100,  # REDUCED from 1000 to 100 Hz for better motor driver compatibility
            board_mode=GPIO.BOARD
    ):
        print(f"Jetson Model: {GPIO.model}")
        print(f"GPIO Mode: {GPIO.BOARD}")
        GPIO.setwarnings(False)

        self.rpwm_pin = rpwm_pin
        self.lpwm_pin = lpwm_pin
        self.R_EN = R_EN
        self.L_EN = L_EN
        self.pwm_freq = pwm_freq

        # Set board mode
        GPIO.setmode(board_mode)
        mode = GPIO.getmode()
        print(f"Current GPIO mode: {mode}")

        # Setup Enable pins FIRST (and keep them HIGH)
        GPIO.setup(self.R_EN, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.L_EN, GPIO.OUT, initial=GPIO.HIGH)
        print(f"Enable pins {R_EN} and {L_EN} set to HIGH")

        # Setup PWM pins
        GPIO.setup(self.rpwm_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.lpwm_pin, GPIO.OUT, initial=GPIO.LOW)
        print(f"PWM pins {rpwm_pin} and {lpwm_pin} initialized")

        # Create PWM objects
        self.r_pwm = GPIO.PWM(self.rpwm_pin, self.pwm_freq)
        self.l_pwm = GPIO.PWM(self.lpwm_pin, self.pwm_freq)

        # Start PWM at 0% duty cycle
        self.r_pwm.start(0)
        self.l_pwm.start(0)
        print(f"PWM started at {pwm_freq} Hz")

        self.current_speed = 0
        self.direction = "stop"

    # ----------------------------- CONTROL METHODS -----------------------------

    def set_speed(self, duty):
        """Update stored duty cycle (0–100)"""
        duty = max(0, min(100, duty))
        self.current_speed = duty

    def accelerate(self, target_duty, ramp_time=2.0):
        """
        Gradually accelerate to target duty cycle

        Args:
            target_duty: Target speed (0-100)
            ramp_time: Time in seconds to reach target speed
        """
        target_duty = max(0, min(100, target_duty))

        # Ensure we're starting fresh
        self.r_pwm.ChangeDutyCycle(0)
        self.l_pwm.ChangeDutyCycle(0)

        # Calculate step parameters
        steps = 50
        step_delay = ramp_time / steps
        duty_increment = target_duty / steps

        print(f"Accelerating to {target_duty}% over {ramp_time}s")

        for step in range(steps + 1):
            current_duty = duty_increment * step
            self.r_pwm.ChangeDutyCycle(current_duty)
            self.current_speed = current_duty

            if step % 10 == 0:  # Print every 10th step
                print(f"Speed: {current_duty:.1f}%")

            time.sleep(step_delay)

        self.direction = "forward"
        print(f"Acceleration complete: {target_duty}%")

    def forward(self, duty=50):
        """
        Drive forward at specified duty cycle

        Args:
            duty: Speed (0-100), default 50%
        """
        duty = max(0, min(100, duty))

        print(f"Forward at {duty}%")

        # Ensure enables are HIGH
        GPIO.output(self.R_EN, GPIO.HIGH)
        GPIO.output(self.L_EN, GPIO.HIGH)

        # Forward: RPWM active, LPWM off
        self.r_pwm.ChangeDutyCycle(duty)
        self.l_pwm.ChangeDutyCycle(0)

        self.direction = "forward"
        self.current_speed = duty

    def reverse(self, duty=50):
        """
        Drive in reverse at specified duty cycle

        Args:
            duty: Speed (0-100), default 50%
        """
        duty = max(0, min(100, duty))

        print(f"Reverse at {duty}%")

        # Ensure enables are HIGH
        GPIO.output(self.R_EN, GPIO.HIGH)
        GPIO.output(self.L_EN, GPIO.HIGH)

        # Reverse: LPWM active, RPWM off
        self.r_pwm.ChangeDutyCycle(0)
        self.l_pwm.ChangeDutyCycle(duty)

        self.direction = "reverse"
        self.current_speed = duty

    def brake(self, ramp_down_time=0.5):
        """
        Soft brake by ramping down speed

        Args:
            ramp_down_time: Time in seconds to brake
        """
        if self.direction not in ["forward", "reverse"]:
            return  # Nothing to brake

        print(f"Braking from {self.current_speed}%")

        try:
            # Determine which PWM to ramp down
            pwm_to_ramp = self.r_pwm if self.direction == "forward" else self.l_pwm

            # Ramp down in steps
            steps = 20
            step_delay = ramp_down_time / steps

            for i in range(steps, -1, -1):
                duty = (self.current_speed * i) / steps
                try:
                    pwm_to_ramp.ChangeDutyCycle(duty)
                except Exception as e:
                    print(f"Error during brake: {e}")
                time.sleep(step_delay)

        except Exception as e:
            print(f"Unexpected error in brake(): {e}")

        # Full stop
        self.stop()
        print("Brake complete")

    def stop(self):
        """Immediately stop all PWM outputs"""
        try:
            self.r_pwm.ChangeDutyCycle(0)
            self.l_pwm.ChangeDutyCycle(0)
        except Exception as e:
            print(f"Error stopping PWM: {e}")

        self.direction = "stop"
        self.current_speed = 0
        print("Motor stopped")

    def test_pwm_output(self, duration=5):
        """
        Test PWM output by cycling duty cycle
        Useful for debugging

        Args:
            duration: Test duration in seconds
        """
        print(f"\n=== Testing PWM Output for {duration}s ===")
        print(f"RPWM Pin: {self.rpwm_pin}, LPWM Pin: {self.lpwm_pin}")
        print(f"R_EN Pin: {self.R_EN}, L_EN Pin: {self.L_EN}")
        print("Watch for motor movement or measure pins with multimeter/oscilloscope")

        # Ensure enables are HIGH
        GPIO.output(self.R_EN, GPIO.HIGH)
        GPIO.output(self.L_EN, GPIO.HIGH)

        start_time = time.time()
        duty = 0
        increasing = True

        try:
            while (time.time() - start_time) < duration:
                # Alternate between RPWM and LPWM
                if (time.time() - start_time) % 2 < 1:
                    # Test forward (RPWM)
                    self.r_pwm.ChangeDutyCycle(duty)
                    self.l_pwm.ChangeDutyCycle(0)
                    print(f"RPWM: {duty}%, LPWM: 0%")
                else:
                    # Test reverse (LPWM)
                    self.r_pwm.ChangeDutyCycle(0)
                    self.l_pwm.ChangeDutyCycle(duty)
                    print(f"RPWM: 0%, LPWM: {duty}%")

                # Ramp duty cycle up and down
                if increasing:
                    duty += 10
                    if duty >= 100:
                        increasing = False
                else:
                    duty -= 10
                    if duty <= RI 0:
                        increasing = True

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nTest interrupted")
        finally:
            self.stop()
            print("=== Test Complete ===\n")

    # ----------------------------- CLEANUP -----------------------------

    def cleanup(self):
        """Clean shutdown of motor driver"""
        print("Cleaning up motor driver...")

        # Stop motors
        try:
            self.stop()
        except Exception as e:
            print(f"Error during stop in cleanup: {e}")

        # Stop PWM
        for pwm_attr in ["r_pwm", "l_pwm"]:
            pwm = getattr(self, pwm_attr, None)
            if pwm:
                try:
                    pwm.stop()
                    print(f"{pwm_attr} stopped")
                except Exception as e:
                    print(f"Error stopping {pwm_attr}: {e}")

        # Disable motor driver
        try:
            GPIO.output(self.R_EN, GPIO.LOW)
            GPIO.output(self.L_EN, GPIO.LOW)
            print("Enable pins set to LOW")
        except Exception as e:
            print(f"Error setting enable pins: {e}")

        # Cleanup GPIO
        try:
            GPIO.cleanup()
            print("GPIO cleanup complete")
        except OSError as e:
            if e.errno == 9:
                print("Ignoring OSError 9 during GPIO.cleanup()")
            else:
                raise
        except Exception as e:
            print(f"Error during GPIO cleanup: {e}")


# ----------------------------- TEST FUNCTION -----------------------------

def test_motor_driver():
    """Standalone test function"""
    print("\n" + "=" * 50)
    print("HW-039 Motor Driver Test")
    print("=" * 50 + "\n")

    motor = MotorDriverHW039(pwm_freq=100)  # Use 100 Hz

    try:
        # Test 1: PWM output test
        motor.test_pwm_output(duration=10)

        # Test 2: Acceleration
        print("\n--- Test: Acceleration ---")
        motor.accelerate(60, ramp_time=3)
        time.sleep(2)

        # Test 3: Forward
        print("\n--- Test: Forward 70% ---")
        motor.forward(70)
        time.sleep(3)

        # Test 4: Brake
        print("\n--- Test: Brake ---")
        motor.brake(ramp_down_time=1)
        time.sleep(1)

        # Test 5: Reverse
        print("\n--- Test: Reverse 60% ---")
        motor.reverse(60)
        time.sleep(3)

        # Test 6: Stop
        print("\n--- Test: Stop ---")
        motor.stop()
        time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    finally:
        motor.cleanup()
        print("\nTest complete!")


if __name__ == "__main__":
    test_motor_driver()