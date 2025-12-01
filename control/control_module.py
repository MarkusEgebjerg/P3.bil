import Jetson.GPIO as GPIO
import time
import cv2

class ControlModule:
    def __init__(self, pwm_pin=33, pwm_freq=50, board_mode=GPIO.BOARD):
        self.pwm_pin = pwm_pin
        self.pwm_freq = pwm_freq
        self._setup_gpio(board_mode)

    def _setup_gpio(self, board_mode):
        GPIO.setmode(board_mode)
        GPIO.setup(self.pwm_pin, GPIO.OUT, initial=GPIO.LOW)
        self.pwm = GPIO.PWM(self.pwm_pin, self.pwm_freq)
        self.pwm.start(0.0)
        print()


    def servo_control(self, duty):
        self.pwm.ChangeDutyCycle(duty)
        print("PWM running. Press CTRL+C to exit.")
        val = 150
        incr = 5
        try:
                while True:
                    time.sleep(0.25)
                    if val >= 200:
                        incr = -incr
                    if val <= 100:
                        incr = -incr
                    val += incr
                    self.pwm.ChangeDutyCycle(val)
        finally:
            self.pwm.stop()
            # GPIO.JETSON_INFO is a dictionary, do not call it like a function ()
            print(GPIO.JETSON_INFO)
            GPIO.cleanup()



    def show_debug(self, img, window_name="debug"):
        if img is None:
            return
        cv2.imshow(window_name, img)
        cv2.waitKey(1)

    def cleanup(self):
        try:
            self.pwm.stop()
        except Exception:
            pass
        GPIO.cleanup()
        cv2.destroyAllWindows()
