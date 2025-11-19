import cv2
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import math


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

    # ------------ Perception helpers ------------

    def get_frame(self):
        frameset = self.pipe.wait_for_frames()
        frames = self.align.process(frameset)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        return depth_frame, color_frame

    def update_intrinsics(self, depth_frame):
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

    def color_space_conversion(self, color_frame):
        color_array = np.asanyarray(color_frame.get_data())
        frame_HSV = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)
        return frame_HSV, color_array

    def Cropping(self,frame_HSV, color_array):
        # ---- CROP TO BOTTOM HALF ----
        h, w = frame_HSV.shape[:2]
        y_offset = h // 2  # start row of bottom half

        frame_HSV_crop = frame_HSV[y_offset:h, 0:w]
        color_crop = color_array[y_offset:h, 0:w]
        return frame_HSV_crop, color_crop, y_offset

    def mask_clean(self, mask):
        mask_Open = cv2.erode(mask, self.kernel, iterations=3)
        mask_Close = cv2.dilate(mask_Open, self.kernel, iterations=10)
        return mask_Close

    def color_detector(self, frame_HSV):
        frame_threshold_Y = cv2.inRange(frame_HSV, self.lowerYellow, self.upperYellow)
        frame_threshold_B = cv2.inRange(frame_HSV, self.lowerBlue, self.upperBlue)

        clean_mask_y = self.mask_clean(frame_threshold_Y)
        clean_mask_b = self.mask_clean(frame_threshold_B)
        return clean_mask_y, clean_mask_b

    def contour_finder(self, clean_mask_y, clean_mask_b):
        contours_b, _ = cv2.findContours(clean_mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_y, _ = cv2.findContours(clean_mask_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_y, contours_b

    def contour_center_finder(self, contours_y, contours_b, color_array):
        # fresh per call
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

    def contour_control(self, contour_centers, color_array):
        cone_positions = []
        for i in range(0, len(contour_centers)):
            p = 0
            for j in range(0, len(contour_centers)):

                if (
                    i != j
                    and -self.afvig < (contour_centers[i][0] - contour_centers[j][0]) < self.afvig
                    and contour_centers[i][1] < contour_centers[j][1]
                ):
                    cone_positions.append(contour_centers[i])

                if (
                    i != j
                    and (
                        -self.afvig > (contour_centers[i][0] - contour_centers[j][0])
                        or self.afvig < (contour_centers[i][0] - contour_centers[j][0])
                    )
                ):
                    p = p + 1
                    if p >= len(contour_centers) - 1:
                        cone_positions.append(contour_centers[i])
        return cone_positions, color_array



    def world_positioning(self, cone_positions, depth_frame, depth_intrin, color_array):
        """
        Convert image-space cone centers to world coordinates.
        Returns:
            color_array_with_text, world_cones
        where world_cones is a list of (u, v, X, Y, Z, color)
        """
        world_cones = []

        for i in range(0, len(cone_positions)):
            u = float(cone_positions[i][0])  # pixel x
            v = float(cone_positions[i][1])  # pixel y
            depth_m = float(depth_frame.get_distance(int(u), int(v)))
            if depth_m <= 0:
                continue

            X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth_m)
            X = round(X, 2)
            Y = round(Y, 2)
            Z = round(Z, 2)
            color = cone_positions[i][2]

            world_cones.append((u, v, X, Y, Z, color))

            cv2.circle(color_array, (int(u), int(v)), 4, (255, 255, 255), -1)
            cv2.putText(
                color_array,
                f" c:{color} coo:[{X}, {Z}]",
                (int(u), int(v)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 0, 255)
            )

        return color_array, world_cones

    # ------------ Perception main step ------------

    def run(self, logic_module):
        depth_frame, color_frame = self.get_frame()
        if depth_frame is None:
            return True

        depth_intrin = self.update_intrinsics(depth_frame)
        depth_frame = self.spatial_filter(depth_frame)
        frame_HSV, color_array = self.color_space_conversion(color_frame)
        frame_HSV_crop, color_crop, y_offset = self.Cropping(frame_HSV,color_array)
        clean_mask_y, clean_mask_b = self.color_detector(frame_HSV_crop)
        contours_y, contours_b = self.contour_finder(clean_mask_y, clean_mask_b)
        contour_centers_crop = self.contour_center_finder(contours_y, contours_b, color_crop)

        # Shift contour centers back to full-image coordinates
        contour_centers = [
            (cx, cy + y_offset, col) for (cx, cy, col) in contour_centers_crop
        ]

        cone_positions, _ = self.contour_control(contour_centers, color_array)
        world_img, world_cones = self.world_positioning(cone_positions, depth_frame, depth_intrin, color_array)

        # Logic module: pair cones, compute midpoints, draw them
        output_img, midpoints_world = logic_module.update(world_cones, world_img)

        cv.imshow("thresh", output_img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return False

        return True

    def shutdown(self):
        self.pipe.stop()
        cv.destroyAllWindows()


class Logic_Module:
    def update(self, world_cones, img):
        img, midpoints_world = self.pair_and_draw_midpoints(world_cones, img)
        # later you can return midpoints_world to motor control
        return img, midpoints_world

    def pair_and_draw_midpoints(self, world_cones, img):
        # Separate cones by color
        blue_cones = [c for c in world_cones if c[5] == "Blue"]
        yellow_cones = [c for c in world_cones if c[5] == "Yellow"]

        # Sort each list by forward distance Z (index 4)
        blue_cones.sort(key=lambda c: c[4])
        yellow_cones.sort(key=lambda c: c[4])

        n_pairs = min(len(blue_cones), len(yellow_cones))
        midpoints_world = []  # (midX, midZ)

        for i in range(n_pairs):
            ub, vb, Xb, Yb, Zb, cb = blue_cones[i]
            uy, vy, Xy, Yy, Zy, cy = yellow_cones[i]

            # Midpoint in world coordinates (X/Z plane)
            midX = (Xb + Xy) / 2.0
            midZ = (Zb + Zy) / 2.0
            midpoints_world.append((midX, midZ))

            # Midpoint in image space for drawing
            mid_u = (ub + uy) / 2.0
            mid_v = (vb + vy) / 2.0

            cv2.circle(img, (int(mid_u), int(mid_v)), 6, (0, 255, 0), -1)
            cv2.putText(
                img,
                "MID",
                (int(mid_u) + 5, int(mid_v)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        return img, midpoints_world


def main():
    try:
        perception = Perception_Module()
    except RuntimeError as e:
        print(e)
        return

    logic = Logic_Module()

    try:
        while True:
            keep_running = perception.run(logic)
            if not keep_running:
                break
    finally:
        perception.shutdown()


if __name__ == '__main__':
    main()
