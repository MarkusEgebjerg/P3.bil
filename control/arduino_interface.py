import serial
import time

class ArduinoInterface:
    def __init__(self, port="/dev/ttyACM0", baud=115200):
        self.ser = serial.Serial(port, baud)
        time.sleep(2)  # wait for Arduino reset

    def send(self, steering, speed):
        """
        steering: -30 to +30 degrees
        speed: 0–255 (or -255 to +255)
        """
        steering = int(steering*10) # to get a dicimal to send and will angle/10
        speed = int(speed*10) # to get a dicimal to send and will speed/10
        msg = f"{steering},{speed}\n"
        self.ser.write(msg.encode())

    def close(self):
        self.ser.close()
