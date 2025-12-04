import cv2
import numpy as np
import pyrealsense2 as rs
import RPi.GPIO as GPIO
import time


output_pins = {
    'JETSON_XAVIER': 18,
    'JETSON_NANO': 33,
    'JETSON_NX': 33,
    'CLARA_AGX_XAVIER': 18,
    'JETSON_TX2_NX': 32,
    'JETSON_ORIN': 18,
    'JETSON_ORIN_NX': 33,
    'JETSON_ORIN_NANO': 33
}
output_pin = output_pins.get(GPIO.model, None)
if output_pin is None:
    raise Exception('PWM not supported on this board')

def run_pwm_test(pin):
    print(f"Starting PWM test on pin {pin}. Press Ctrl+C to stop.")
    # Set pin as an output pin with optional initial state of HIGH
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)
    p = GPIO.PWM(pin, 50)
    val = 25
    incr = 5
    p.start(val)
    try:
        while True:
            time.sleep(0.25)
            if val >= 100:
                incr = -incr
            if val <= 0:
                incr = -incr
            val += incr
            p.ChangeDutyCycle(val)
            print(f"Duty Cycle: {val}")
    except KeyboardInterrupt:
        print("Test stopped by user")
    finally:
        p.stop()
        GPIO.cleanup()


class Perception_Module:
    def __init__(self):
        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        self.res_x = 1280
        self.res_y = 720
        self.fps = 30

        self.cfg.enable_stream(rs.stream.color, self.res_x, self.res_y, rs.format.bgr8, self.fps)
        self.cfg.enable_stream(rs.stream.depth, self.res_x, self.res_y, rs.format.z16, self.fps)
        self.align = rs.align(rs.stream.color)


        self.stream = self.pipe.start(self.cfg)
        self.depth_sensor = self.stream.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.depth_intrin = None

        self.lowerYellow = np.array([20, 110, 90])
        self.upperYellow = np.array([33, 255, 255])
        self.lowerBlue = np.array([105, 120, 60])
        self.upperBlue = np.array([135, 255, 255])

        self.kernel = np.ones((2, 3))
        self.afvig = 25

        self.spatial = rs.spatial_filter()

        self.logic = Logic_module()

    def get_frame(self):
        frameset = self.pipe.wait_for_frames()
        frames = self.align.process(frameset)

        # Tager dybde- og farve billeddet af samlingen "frames"
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        return depth_frame, color_frame

    def update_intrinsics(self,depth_frame):
        if self.depth_intrin is None:
            self.depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        return self.depth_intrin

    def spatial_filter(self, depth_frame):

        depth_frame = self.spatial.process(depth_frame)

        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 1)
        self.spatial.set_option(rs.option.filter_smooth_delta, 50)
        depth_frame = self.spatial.process(depth_frame)

        self.spatial.set_option(rs.option.holes_fill, 0)
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()

        return depth_frame

    def color_space_conversion (self, color_frame):
        color_array = np.asanyarray(color_frame.get_data())
        res_y, res_x = color_array.shape[:2]
        color_array = color_array[res_y//2:res_y, 0:res_x]
        frame_HSV = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)
        return frame_HSV, color_array

    def mask_clean(self, mask):
        mask_Open = cv2.erode(mask, self.kernel, iterations=2)
        mask_Close = cv2.dilate(mask_Open, self.kernel, iterations=10)
        return mask_Close

    def color_detector(self, frame_HSV):
        frame_threshold_Y = cv2.inRange(frame_HSV, self.lowerYellow, self.upperYellow)
        frame_threshold_B = cv2.inRange(frame_HSV, self.lowerBlue, self.upperBlue)

        clean_mask_y = self.mask_clean(frame_threshold_Y)
        clean_mask_b = self.mask_clean(frame_threshold_B)
        return clean_mask_y, clean_mask_b


    def contour_finder(self,clean_mask_y, clean_mask_b):
        contours_b, _ = cv2.findContours(clean_mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_y, _ = cv2.findContours(clean_mask_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_y, contours_b


    def contour_center_finder(self, contours_y, contours_b,color_array): #finds contour center and draws contour
        self.contour_centers = []
        for i in range(2):
            list = contours_b
            if i == 1:
                list = contours_y

            for contour in list:
                area = cv2.contourArea(contour)
                if area < 30:
                    continue

                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if i == 1:
                    self.current_center = (cx, cy, "Yellow")
                    self.contour_centers.append(self.current_center)
                else:
                    self.current_center = (cx, cy, "Blue")
                    self.contour_centers.append(self.current_center)

                # Draw the contour
                if len(approx) > 3:
                    cv2.drawContours(color_array, [approx], 0, (255, 0, 0), 5)
        return self.contour_centers

    def contour_control(self, contour_centers, color_array):
        self.cone_positions = []
        # Loops through alle contour centers. checks if two of them are less than "Afvig" pixels away horizontally (x).
        # if there are, the center highest up (biggest y) is added to cone_positions.
        #also draws a circle of each cone position.
        for i in range(0, len(contour_centers)):
            p = 0
            for j in range(0, len(contour_centers)):

                if i != j and -self.afvig < (contour_centers[i][0] - contour_centers[j][0]) < self.afvig and contour_centers[i][
                    1] < \
                        contour_centers[j][1]:
                    self.cone_positions.append(contour_centers[i])  # øverste contour gemmes i cone_positions

                if i != j and -self.afvig > (contour_centers[i][0] - contour_centers[j][0]) or self.afvig < (
                        contour_centers[i][0] - contour_centers[j][0]):
                    p = p + 1
                    if p >= len(contour_centers) - 1:
                        #cv2.circle(color_array, (contour_centers[i][0], contour_centers[i][1]), 4, (255, 255, 255), -1)
                        self.cone_positions.append(contour_centers[i])  # contour gemmes hvis den står alane
                if len(contour_centers) == 1:
                    self.cone_positions.append(contour_centers[i])  # contour gemmes hvis den står alane
        return self.cone_positions, color_array

    def world_positioning(self, cone_positions, depth_frame,depth_intrin,color_array):
        world_cones = []
        for i in range(0, len(cone_positions)):
            u = float(cone_positions[i][0])  # pixel x
            v = float(cone_positions[i][1])  # pixel y
            depth_m = float(depth_frame.get_distance(int(u), int(v)+360))
            if depth_m <= 0:
                continue

            X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth_m)
            X = round(X, 2)
            Y = round(Y, 2)
            Z = round(Z, 2)

            color = cone_positions[i][2]

            if Z < 6:

                world_cones.append((X, Z, color, u, v))
                cv2.circle(color_array, (cone_positions[i][0], cone_positions[i][1]), 4, (255, 255, 255), -1)
                cv2.putText(color_array, f" c: {(cone_positions[i][2])} coo: {[X, Z]}", (int(u), int(v)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return  world_cones, color_array

    def run(self):
        depth_frame, color_frame = self.get_frame()
        if depth_frame is None:
            return True

        depth_intrin = self.update_intrinsics(depth_frame)
        depth_frame = self.spatial_filter(depth_frame)
        frame_HSV, color_array = self.color_space_conversion(color_frame)
        clean_mask_y, clean_mask_b = self.color_detector(frame_HSV)
        contours_y, contours_b = self.contour_finder(clean_mask_y, clean_mask_b)
        contour_centers = self.contour_center_finder(contours_y, contours_b, color_array)
        cone_positions, color_array = self.contour_control(contour_centers, color_array)
        world_pos, color_array= self.world_positioning(cone_positions, depth_frame,depth_intrin, color_array)
        blue_cones, yellow_cones = self.logic.cone_sorting(world_pos)
        midpoints = self.logic.cone_midpoints(blue_cones, yellow_cones, color_array)

        target = self.logic.Interpolation(midpoints)
        if target is not None:
            steering = self.logic.streering_angle(target)
            # for now: just print it so you see it works
            print(f"Target: {target}, steering angle: {steering:.3f} deg")
        else:
            print("No target found (no midpoints)")

        #cv2.imshow("thresh", color_array)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        return True

    def shutdown(self):
        self.pipe.stop()
        cv2.destroyAllWindows()

class Logic_module:
    def __init__(self):
        self.l = 0.5
        self.WD = 0.3
        self.max_p = 4

    def cone_sorting(self, world_cones):
        blue_cones = [c for c in world_cones if c[2] == "Blue"]
        yellow_cones = [c for c in world_cones if c[2] == "Yellow"]

        blue_cones.sort(key=lambda c: c[1])
        yellow_cones.sort(key=lambda c: c[1])
        return blue_cones, yellow_cones

    def cone_midpoints(self, blue_cones, yellow_cones, color_array):
        midpoints = []
        n = min(len(blue_cones), len(yellow_cones), self.max_p)
        for i in range(n):
            bx, bz, _, bu, bv = blue_cones[i]
            yx, yz, _, yu, yv = yellow_cones[i]

            x = (bx+yx) / 2
            z = (bz+yz) / 2

            u = int ((bu+yu)/2)
            v = int ((bv+yv) / 2)

            cv2.circle(color_array, (u,v), 4, (0, 255, 0), -1)
            cv2.putText(color_array, f" coo: {[round(x,2), round(z,2)]}", (u,v), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            midpoints.append((x,z))
        return midpoints

    def Interpolation(self, midpoints):


        if len(midpoints) > 0:
            x = sorted([x[0] for x in midpoints])
            z = sorted([x[1] for x in midpoints])
            for i in range(len(z)): z[i] = z[i]+self.WD
            def length(x, z):
                len = np.sqrt(x**2 + z**2)
                return len

            if length(x[0], z[0]) >= self.l:

                s = self.l / (np.sqrt(x[0]**2 + z[0] ** 2))
                target = (s*x[0], s*z[0])  # target point for pure pursuit


            elif length(x[0], z[0]) < self.l:
                # uden fugleflugt
                # rest_l = l-length(x[0], z[0])
                # s = rest_l/(np.sqrt(x[0]2 + z[0]2))
                # target  = (c[0][0] + sc[1][0], c[0][1] + sc[1][1]) #target point for pure pursuit

                # fugleflugt
                A = x[1]**2 + z[1]**2
                B = 2*(x[0] * x[1] + z[0] * z[1])
                C = x[0]**2 + z[0]*2 - self.l**2
                D = B**2 - (4*A*C)
                if D >= 0:
                    s_plus = (-B + np.sqrt(D)) / (2*A)
                    s_minus = (-B - np.sqrt(D)) / (2*A)
                    positive_s = [s for s in (s_plus, s_minus) if s > 0]
                    s = positive_s
                    target = (np.float64(x[0] + s[0] * x[1]), np.float64(z[0] + s[0] * z[1]))  # target point for pure pursuit



            return target

    def streering_angle(self, target):
        r = self.l**2 / (target[0]*2)
        steeringangle = np.arctan(self.WD/r)
        steeringangle = np.rad2deg(steeringangle)

        return steeringangle



def main():
    # Pin Setup:
    # Board pin-numbering scheme
    GPIO.setmode(GPIO.BOARD)

    # run the standalone PWM test
    print("run pwm test")
    run_pwm_test(output_pin)

    try:
        perception= Perception_Module()
    except RuntimeError as e:
        print(e)
        return

    try:
        while True:
            keep_running = perception.run()
            if not keep_running:
                break

    finally:
        perception.shutdown()


if __name__ == '__main__':
    main()