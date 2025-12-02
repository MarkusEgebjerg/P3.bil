import Jetson.GPIO as GPIO
import time

from sympy.codegen import Print


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
        print(GPIO.BOARD)

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
        # Soft brake: ramp down speed
        for d in range(self.current_speed, -1, -10):
            if self.direction == "forward":
                self.r_pwm.ChangeDutyCycle(d)
            elif self.direction == "reverse":
                self.l_pwm.ChangeDutyCycle(d)
            time.sleep(0.01)

        self.stop()

    def stop(self):
        # Stop PWM output
        self.r_pwm.ChangeDutyCycle(0)
        self.l_pwm.ChangeDutyCycle(0)
        self.direction = "stop"
        self.current_speed = 0

    # ----------------------------- CLEANUP ------------------------------------

    def cleanup(self):
        self.stop()
        try:
            self.r_pwm.stop()
            self.l_pwm.stop()
        except:
            pass

        GPIO.cleanup()
