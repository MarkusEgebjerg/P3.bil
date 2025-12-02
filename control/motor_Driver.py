import Jetson.GPIO as GPIO
import time


class MotorDriverHW039:
    """
    HW-039 / BTS7960 High-Power Motor Driver Controller
    using Jetson.GPIO
    """

    def __init__(
        self,
        rpwm_pin=32,   # PWM pin
        lpwm_pin=33,   # PWM pin
        R_EN=29,       # digital output
        L_EN=31,       # digital output
        pwm_freq=1000,
        board_mode=GPIO.BOARD

    ):
        print(f"Jetson Model: {GPIO.model}")
        print(GPIO.BOARD)
        GPIO.setwarnings(False)
        self.rpwm_pin = rpwm_pin
        self.lpwm_pin = lpwm_pin
        self.R_EN = R_EN
        self.L_EN = L_EN
        self.pwm_freq = pwm_freq

        GPIO.setmode(board_mode)
        mode = GPIO.getmode()
        print(mode)
                

        # Enable pins
        GPIO.setup(self.R_EN, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.L_EN, GPIO.OUT, initial=GPIO.HIGH)

        # PWM pins
        GPIO.setup(self.rpwm_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.lpwm_pin, GPIO.OUT, initial=GPIO.LOW)

        # PWM objects (each side has its own PWM)
        self.r_pwm = GPIO.PWM(self.rpwm_pin, self.pwm_freq)
        self.l_pwm = GPIO.PWM(self.lpwm_pin, self.pwm_freq)

        self.r_pwm.start(0)
        self.l_pwm.start(0)

        self.current_speed = 0
        self.direction = "stop"

    # ----------------------------- CONTROL METHODS -----------------------------

    def set_speed(self, duty):
        """
        Update stored duty cycle (0–100), but actual output
        depends on direction method.
        """
        duty = max(0, min(100, duty))
        self.current_speed = duty

        # Direction method sets which PWM is active.

    def accelerate(self, duty):
        self.set_speed(duty)
        self.r_pwm.ChangeDutyCycle(0)
        self.l_pwm.ChangeDutyCycle(0)
        for i in range(duty):
            self.r_pwm.ChangeDutyCycle(i)
            self.current_speed = i
            print(i)
        self.direction = "forward"


    def forward(self, duty=100):
        self.stop()

        GPIO.output(self.R_EN, GPIO.HIGH)
        GPIO.output(self.L_EN, GPIO.HIGH)

        # Forward: RPWM high PWM, LPWM low
        self.r_pwm.ChangeDutyCycle(duty)
        self.l_pwm.ChangeDutyCycle(0)

        self.direction = "forward"
        self.current_speed = duty

    def reverse(self, duty=100):
        self.stop()

        GPIO.output(self.R_EN, GPIO.HIGH)
        GPIO.output(self.L_EN, GPIO.HIGH)

        # Reverse: LPWM high PWM, RPWM low
        self.r_pwm.ChangeDutyCycle(0)
        self.l_pwm.ChangeDutyCycle(duty)

        self.direction = "reverse"
        self.current_speed = duty

    def brake(self):
        """Soft brake by ramping down speed, safely handling PWM state."""
        if self.direction not in ["forward", "reverse"]:
            return  # nothing to brake

        try:
            # Determine which PWM to ramp down
            pwm_to_ramp = self.r_pwm if self.direction == "forward" else self.l_pwm

            # Ramp down speed safely
            for d in range(self.current_speed, -1, -10):
                try:
                    pwm_to_ramp.ChangeDutyCycle(d)
                except Exception as e:
                    print(f"Ignoring ChangeDutyCycle error during brake: {e}")
                time.sleep(0.01)

        except Exception as e:
            print(f"Ignoring unexpected error in brake(): {e}")

        # Ensure motors are fully stopped
        self.stop()

    def stop(self):
        """Safely stop PWM outputs without crashing if PWM is already stopped."""
        for pwm_attr in ["r_pwm", "l_pwm"]:
            pwm = getattr(self, pwm_attr, None)
            if pwm:
                try:
                    pwm.ChangeDutyCycle(0)
                except Exception as e:
                    print(f"Ignoring ChangeDutyCycle error for {pwm_attr} in stop(): {e}")

        self.direction = "stop"
        self.current_speed = 0

    #  CLEANUP

    def cleanup(self):
        # Stop motors
        try:
            self.stop()
        except Exception as e:
            print(f"Ignoring stop() error during cleanup: {e}")

        # Stop PWM
        for pwm_attr in ["r_pwm", "l_pwm"]:
            pwm = getattr(self, pwm_attr, None)
            if pwm:
                try:
                    pwm.stop()
                except Exception as e:
                    print(f"Ignoring PWM stop error for {pwm_attr}: {e}")

        # Cleanup GPIO
        try:
            GPIO.cleanup()
        except OSError as e:
            if e.errno == 9:
                # Bad file descriptor; safe to ignore in Docker/PWM cases
                print("Ignoring OSError 9 during GPIO.cleanup()")
            else:
                raise
        except Exception as e:
            print(f"Ignoring unexpected GPIO cleanup error: {e}")