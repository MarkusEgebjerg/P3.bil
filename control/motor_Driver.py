import Jetson.GPIO as GPIO
import time


class MotorDriverHW039:
    """
    HW-039 / BTS7960 High-Power Motor Driver Controller
    using Jetson.GPIO
    """

    def __init__(
        self,
        rpwm_pin=32,
        lpwm_pin=33,
        enable_pin=33,       # If unused: set to None
        pwm_freq=1000,
        board_mode=GPIO.BOARD
    ):
        self.rpwm_pin = rpwm_pin
        self.lpwm_pin = lpwm_pin
        self.enable_pin = enable_pin
        self.pwm_freq = pwm_freq

        GPIO.setmode(board_mode)

        # Direction pins
        GPIO.setup(self.rpwm_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.lpwm_pin, GPIO.OUT, initial=GPIO.LOW)

        # Optional enable pin + PWM
        if self.enable_pin is not None:
            GPIO.setup(self.enable_pin, GPIO.OUT, initial=GPIO.LOW)
            self.pwm = GPIO.PWM(self.enable_pin, self.pwm_freq)
            self.pwm.start(0)
        else:
            self.pwm = None

        self.current_speed = 0
        self.direction = "stop"

    # ----------------------------- CONTROL METHODS -----------------------------

    def set_speed(self, duty):
        """
        Set speed using PWM (0-100).
        """
        duty = max(0, min(100, duty))
        self.current_speed = duty

        if self.pwm is not None:
            self.pwm.ChangeDutyCycle(duty)

    def forward(self, duty=100):
        """
        Rotate motor forward.
        """
        self.stop()  # always stop before direction change
        GPIO.output(self.rpwm_pin, GPIO.HIGH)
        GPIO.output(self.lpwm_pin, GPIO.LOW)

        self.set_speed(duty)
        self.direction = "forward"

    def reverse(self, duty=100):
        """
        Rotate motor backward.
        """
        self.stop()
        GPIO.output(self.rpwm_pin, GPIO.LOW)
        GPIO.output(self.lpwm_pin, GPIO.HIGH)

        self.set_speed(duty)
        self.direction = "reverse"

    def brake(self):
        """
        Electronic braking (both sides HIGH).
        """
        GPIO.output(self.rpwm_pin, GPIO.HIGH)
        GPIO.output(self.lpwm_pin, GPIO.HIGH)
        self.set_speed(0)
        self.direction = "brake"

    def stop(self):
        """
        Disable both direction pins.
        """
        GPIO.output(self.rpwm_pin, GPIO.LOW)
        GPIO.output(self.lpwm_pin, GPIO.LOW)
        if self.pwm is not None:
            self.pwm.ChangeDutyCycle(0)
        self.direction = "stop"

    # ----------------------------- CLEANUP ------------------------------------

    def cleanup(self):
        self.stop()
        if self.pwm is not None:
            try:
                self.pwm.stop()
            except:
                pass
        GPIO.cleanup()

