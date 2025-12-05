import serial
import time

class ArduinoInterface:
    def __init__(self, port="/dev/ttyACM0", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)

    def send(self, steering, speed):
        steering = int(steering * 10)
        speed = int(speed * 10)

        msg = f"{steering},{speed}\n"
        self.ser.write(msg.encode())

        #READ FEEDBACK
        try:
            line = self.ser.readline().decode().strip()
            if line:
                print("Arduino:", line)
                return line
        except:
            pass

        return None

    def close(self):
        self.ser.close()
