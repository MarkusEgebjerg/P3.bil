from perception.perception_module import PerceptionModule
from logic.logic_module import LogicModule
from control.control_module import ControlModule

def main():
    perception = PerceptionModule()
    logic = LogicModule()
    control = ControlModule()

    while True:
        cones_world, img = perception.run()

        blue, yellow = logic.cone_sorting(cones_world)
        midpoints = logic.cone_midpoints(blue, yellow, img)
        target = logic.Interpolation(midpoints)

        if target:
            angle = logic.streering_angle(target)
            control.set_steering(angle)

        control.show_debug(img)

if __name__ == "__main__":
    main()