import cv2
import numpy as np
import pyrealsense2 as rs


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

        self.lowerYellow = np.array([22, 110, 120])
        self.upperYellow = np.array([33, 255, 255])
        self.lowerBlue = np.array([100, 120, 60])
        self.upperBlue = np.array([135, 255, 255])

        self.z_window_size = 5  # how many past Z values to remember
        self.z_histories = {}  # list of recent Z values


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
        color_array = color_array[int(res_y*(1/4)):int(res_y*(3/4)), 0:res_x]
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

    def smooth_z(self, key, new_z: float) -> float:
        """
        Smooth Z per cone key (e.g. per (color, u_bin, v_bin)).
        Keeps a rolling window of the last N values for that cone.
        """
        if new_z <= 0:
            return new_z

        history = self.z_histories.get(key, [])
        history.append(new_z)

        if len(history) > self.z_window_size:
            history.pop(0)

        self.z_histories[key] = history
        return sum(history) / len(history)

    def contour_center_finder(self, contours_y, contours_b, color_array):
        self.contour_centers = []

        for i in range(2):
            contour_list = contours_b if i == 0 else contours_y

            for contour in contour_list:
                area = cv2.contourArea(contour)
                if area < 30:
                    continue

                # Bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cx = int(x + w / 2)
                cy = int(y + h / 2)

                color_label = "Blue" if i == 0 else "Yellow"
                self.current_center = (cx, cy, color_label)
                self.contour_centers.append(self.current_center)

                cv2.rectangle(color_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

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
            depth_m = float(depth_frame.get_distance(int(u), int(v)+180))
            if depth_m <= 0:
                continue

            X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth_m)
            color = cone_positions[i][2]
            key = (color, int(u // 10), int(v // 10))
            Z = self.smooth_z(key,Z)

            X = round(X, 2)
            Y = round(Y, 2)
            Z = round(Z, 2)
            print(f'Z: {Z}')
            print(f'X: {X}')


            if Z < 6:

                world_cones.append((X, Z, color, u, v))
                cv2.circle(color_array, (cone_positions[i][0], cone_positions[i][1]), 4, (255, 255, 255), -1)
                cv2.putText(color_array, f" c: {(cone_positions[i][2])} coo: {[X, Z]}", (int(u), int(v)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
                cv2.line(color_array, ((1280//2),0),((1280//2),720), (255,255,255),2)
                cv2.line(color_array, (0,(720//4)), (1280,(720//4)), (255,255,255),2)

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

        cv2.imshow("thresh", color_array)
        cv2.imshow("Yellow", clean_mask_y)
        cv2.imshow("Blue", clean_mask_b)
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
            for i in range(len(z)): z[i] = z[i]+self.WD # adding wheelbase to z depth value to use in pure pursuit
            def length(x, z):
                len = np.sqrt(x**2 + z**2)
                return len

            if length(x[0], z[0]) >= self.l:

                c = self.l / (np.sqrt(x[0]**2 + z[0] ** 2)) # computing skalar, c
                target = (c*x[0], c*z[0])  # target point for pure pursuit


            elif length(x[0], z[0]) < self.l:
                # uden fugleflugt
                # rest_l = l-length(x[0], z[0])
                # s = rest_l/(np.sqrt(x[0]2 + z[0]2))
                # target  = (c[0][0] + sc[1][0], c[0][1] + sc[1][1]) #target point for pure pursuit

                # fugleflugt________
                a = x[1]-x[0] # defined for easier following computation
                b = z[1]-z[0] # defined for easier following computation

                A = a^2 +b^2
                B = 2(x[0] * a + z[0] * b)
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