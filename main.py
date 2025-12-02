from perception.perception_module import PerceptionModule
from logic.logic_module import LogicModule
from control.control_module import ControlModule
from control.motor_Driver import MotorDriverHW039
import time


def main():
#    perception = PerceptionModule()
#    logic = LogicModule()
#"    control = ControlModule()
    motor = MotorDriverHW039()   # your pin setup is loaded here

    try:
        print("Forward 50%...")
        motor.forward(50)
        time.sleep(2)

        print("Reverse 50%...")
        motor.reverse(50)
        time.sleep(2)

        print("Braking...")
        motor.brake()
        time.sleep(1)

        print("Stopping...")
        motor.stop()
        time.sleep(1)

    except KeyboardInterrupt:
        print("Interrupted")

    finally:
        print("Cleanup GPIO")
        motor.cleanup()
    #while True:
        #cones_world, img = perception.run()

        #blue, yellow = logic.cone_sorting(cones_world)
        #midpoints = logic.cone_midpoints(blue, yellow, img)
        #target = logic.Interpolation(midpoints)

        #if target:
        #    angle = logic.streering_angle(target)
        #    control.set_steering(angle)

        #control.show_debug(img)

if __name__ == "__main__":
    main()