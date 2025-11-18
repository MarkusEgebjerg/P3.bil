import cv2
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import math


class PerceptionModule:
    def __init__(self):
    # ---- Camera & RealSense setup ----
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.res_x = 1280
        self.res_y = 720
        self.fps = 30
        self.fov_x = 87

        self.cfg.enable_stream(rs.stream.color, self.res_x, self.res_y, rs.format.bgr8, self.fps)
        self.cfg.enable_stream(rs.stream.depth, self.res_x, self.res_y, rs.format.z16, self.fps)

        self.stream = self.pipe.start(self.cfg)
        depth_sensor = self.stream.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # ---- Color thresholds & morphology ----
        self.lowerYellow = np.array([20, 110, 90])
        self.upperYellow = np.array([33, 255, 255])

        self.lowerBlue = np.array([105, 120, 60])
        self.upperBlue = np.array([135, 255, 255])

        self.kernel = np.ones((2, 3))
        self.afvig = 25

        self.PxD = self.res_x / self.fov_x
        self.align = rs.align(rs.stream.color)

        self.depth_intrin = None

        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()

    # ---------- Small helper methods ----------

    def get_frames(self):
        frameset = self.pipe.wait_for_frames()
        frames = self.align.process(frameset)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        return depth_frame, color_frame

    def update_intrinsics(self, depth_frame):
        if self.depth_intrin is None:
            self.depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()

    def filter_depth(self, depth_frame):
        depth_frame = self.spatial.process(depth_frame)

        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 1)
        self.spatial.set_option(rs.option.filter_smooth_delta, 50)
        depth_frame = self.spatial.process(depth_frame)

        self.spatial.set_option(rs.option.holes_fill, 0)
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()

        depth_array = np.asanyarray(depth_frame.get_data())
        depth_img = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_array, alpha=0.06),
            cv2.COLORMAP_JET
        )
        return depth_frame, depth_img

    def preprocess_color(self, color_frame):
        color_array = np.asanyarray(color_frame.get_data())
        frame_HSV = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)
        return color_array, frame_HSV

    def masking(self, mask):
        mask_open = cv2.erode(mask, self.kernel, iterations=3)
        mask_close = cv2.dilate(mask_open, self.kernel, iterations=10)
        return mask_close

    def threshold_and_mask(self, frame_HSV):
        frame_threshold_Y = cv2.inRange(frame_HSV, self.lowerYellow, self.upperYellow)
        frame_threshold_B = cv2.inRange(frame_HSV, self.lowerBlue, self.upperBlue)

        mask_y = self.masking(frame_threshold_Y)
        mask_b = self.masking(frame_threshold_B)
        return mask_y, mask_b

    def find_contours(self, mask_y, mask_b):
        contours_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_y, _ = cv2.findContours(mask_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_b, contours_y

    def find_contour_centers(self, contours_b, contours_y, color_array):
        contour_centers = []

        for i in range(2):
            lst = contours_b
            if i == 1:
                lst = contours_y

            for contour in lst:
                area = cv2.contourArea(contour)
                if area < 60:
                    continue

                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                if i == 1:
                    current_center = (cx, cy, "Yellow")
                else:
                    current_center = (cx, cy, "Blue")
                contour_centers.append(current_center)

                if len(approx) > 3:
                    cv2.drawContours(color_array, [approx], 0, (255, 0, 0), 5)

        return contour_centers

    def select_cones(self, contour_centers, color_array):
        cone_positions = []

        for i in range(0, len(contour_centers)):
            p = 0
            for j in range(0, len(contour_centers)):

                if (
                    i != j and
                    -self.afvig < (contour_centers[i][0] - contour_centers[j][0]) < self.afvig and
                    contour_centers[i][1] < contour_centers[j][1]
                ):
                    cone_positions.append(contour_centers[i])

                if (
                    i != j and
                    (-self.afvig > (contour_centers[i][0] - contour_centers[j][0]) or
                     self.afvig < (contour_centers[i][0] - contour_centers[j][0]))
                ):
                    p += 1
                    if p >= len(contour_centers) - 1:
                        cv2.circle(
                            color_array,
                            (contour_centers[i][0], contour_centers[i][1]),
                            4, (255, 255, 255), -1
                        )
                        cone_positions.append(contour_centers[i])

        return cone_positions

    def project_cones(self, cone_positions, depth_frame, color_array):
        world_cones = []

        for cx, cy, ccolor in cone_positions:
            u = float(cx)
            v = float(cy)

            depth_m = float(depth_frame.get_distance(int(u), int(v)))
            if depth_m <= 0:
                continue

            X, Y, Z = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [u, v], depth_m)
            X = round(X, 2)
            Y = round(Y, 2)
            Z = round(Z, 2)

            world_cones.append((X, Y, Z, ccolor))

            cv2.putText(
                color_array,
                f" c: {ccolor} coo: [{X}, {Z}]",
                (int(u) - 250, int(v)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 0, 255)
            )

        return world_cones, color_array

    # ---------- One perception step ----------

    def step(self, logic_module, motor_module):
        depth_frame, color_frame = self.get_frames()
        if depth_frame is None:
            return True  # continue loop

        self.update_intrinsics(depth_frame)

        depth_frame, depth_img = self.filter_depth(depth_frame)
        color_array, frame_HSV = self.preprocess_color(color_frame)
        mask_y, mask_b = self.threshold_and_mask(frame_HSV)
        contours_b, contours_y = self.find_contours(mask_y, mask_b)
        contour_centers = self.find_contour_centers(contours_b, contours_y, color_array)
        cone_positions = self.select_cones(contour_centers, color_array)
        world_cones, color_array = self.project_cones(cone_positions, depth_frame, color_array)

        # Later:
        # decisions = logic_module.update(world_cones)
        # motor_module.update(decisions)

        cv.imshow("thresh", color_array)
        cv.imshow("depth", depth_img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            return False  # stop loop

        return True  # keep running

    def shutdown(self):
        self.pipe.stop()
        cv.destroyAllWindows()


class LogicModule:
    def __init__(self):
        pass

    def update(self, world_cones):
        # TODO: implement path planning / decisions
        return None


class MotorControlModule:
    def __init__(self):
        pass

    def update(self, command):
        # TODO: send commands to motors/servo
        pass


def main():
    try:
        perception = PerceptionModule()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("-> Check that your RealSense is plugged in and not used by another program.")
        return

    logic = LogicModule()
    motor = MotorControlModule()

    try:
        while True:
            keep_running = perception.step(logic, motor)
            if not keep_running:
                break
    finally:
        perception.shutdown()


if __name__ == "__main__":
    main()
