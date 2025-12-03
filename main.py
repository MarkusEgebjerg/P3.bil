from perception.perception_module import PerceptionModule
from logic.logic_module import LogicModule
from control.control_module import ControlModule
from control.motor_Driver import MotorDriverHW039
from control.PWM_Diagnostic_Tool import pinDiagnostic
import time


def main():
#    perception = PerceptionModule()
#    logic = LogicModule()
#    control = ControlModule()
    motor = MotorDriverHW039()   # your pin setup is loaded here
    testPWM = pinDiagnostic()

    try:
        print("Starting...")
        testPWM.main()
        #motor.test_motor_driver()

    except KeyboardInterrupt:
        print("Interrupted")

    finally:
        print("Cleanup GPIO")
        #motor.cleanup()
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