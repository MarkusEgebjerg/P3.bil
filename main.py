from perception.perception_module import PerceptionModule
from logic.logic_module import LogicModule
from control.arduino_interface import ArduinoInterface
import time


def main():
    perception = PerceptionModule()
    logic = LogicModule()
    arduino = ArduinoInterface("/dev/ttyACM0")

    try:
        print("Starting control loop...")

        while True:
            cones_world, img = perception.run()

            blue, yellow = logic.cone_sorting(cones_world)
            midpoints = logic.cone_midpoints(blue, yellow, img)
            target = logic.Interpolation(midpoints)

            angle = logic.steering_angle(target) if target else 0
            speed = 120  # constant speed

            # Send to Arduino
            arduino.send(angle, speed)

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        arduino.close()
        print("Arduino connection closed.")
