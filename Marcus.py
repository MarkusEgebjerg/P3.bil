import cv2
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import math


# -------------------------
# Camera handling (RealSense)
# -------------------------
class RealSenseCam:
    def __init__(self, res_x=1280, res_y=720, fps=30, fov_x=87):
        self.res_x = res_x
        self.res_y = res_y
        self.fps = fps
        self.fov_x = fov_x

        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, self.res_x, self.res_y, rs.format.bgr8, self.fps)
        self.cfg.enable_stream(rs.stream.depth, self.res_x, self.res_y, rs.format.z16, self.fps)

        self.stream = self.pipe.start(self.cfg)
        depth_sensor = self.stream.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.align = rs.align(rs.stream.color)

        # For downstream math (pixels per degree) – kept for compatibility, but not used for angle now.
        self.PxD = self.res_x / self.fov_x

        # NEW: fetch intrinsics for accurate angle computation
        color_profile = self.stream.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intr = color_profile.get_intrinsics()  # has fx, fy, ppx, ppy, width, height

    def get_processed_frames(self):
        """Get aligned frames + apply the same filtering/processing as in the original code."""
        frameset = self.pipe.wait_for_frames()
        frames = self.align.process(frameset)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Same filters, same order and options
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()  # created but not used further (same as original)

        depth_frame = spatial.process(depth_frame)

        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 1)
        spatial.set_option(rs.option.filter_smooth_delta, 50)
        depth_frame = spatial.process(depth_frame)

        spatial.set_option(rs.option.holes_fill, 0)
        depth_frame = spatial.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()

        depth_array = np.asanyarray(depth_frame.get_data())
        depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=0.06), cv2.COLORMAP_JET)

        color_array = np.asanyarray(color_frame.get_data())

        return color_array, depth_frame, depth_img

    def stop(self):
        self.pipe.stop()


# -------------------------
# Perception (masking + contours + cone pairing)
# -------------------------
class ConePerception:
    def __init__(self, res_x=1280, fov_x=87, afvig=25):
        # Thresholds and kernel from original
        self.lowerYellow = np.array([20, 110, 90])
        self.upperYellow = np.array([33, 255, 255])

        self.lowerBlue = np.array([105, 120, 60])
        self.upperBlue = np.array([135, 255, 255])

        self.kernel = np.ones((2, 3))
        self.afvig = afvig

        # For angle math (kept for compatibility if needed elsewhere)
        self.PxD = res_x / fov_x
        self.fov_x = fov_x

    # Original masking function preserved
    def Masking(self, mask):
        mask_Open = cv2.erode(mask, self.kernel, iterations=3)
        mask_Close = cv2.dilate(mask_Open, self.kernel, iterations=10)
        # mask_Open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        # mask_Close = cv2.morphologyEx(mask_Close_1, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask_Close

    def process_color(self, color_array):
        """HSV thresholding, masking and contour extraction (same logic)."""
        frame_HSV = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)
        frame_threshold_Y = cv2.inRange(frame_HSV, self.lowerYellow, self.upperYellow)
        frame_threshold_B = cv2.inRange(frame_HSV, self.lowerBlue, self.upperBlue)

        Masking_Clean_Y = self.Masking(frame_threshold_Y)
        Masking_Clean_B = self.Masking(frame_threshold_B)

        contours_b, _ = cv2.findContours(Masking_Clean_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_y, _ = cv2.findContours(Masking_Clean_Y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Build centers and draw approximated contours (as in original)
        contour_centers = []
        for i in range(2):
            list_contours = contours_b if i == 0 else contours_y
            for contour in list_contours:
                area = cv2.contourArea(contour)
                if area < 60:
                    continue

                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                current_center = (cx, cy, "Blue" if i == 0 else "Yellow")
                contour_centers.append(current_center)

                if len(approx) > 3:
                    cv2.drawContours(color_array, [approx], 0, (255, 0, 0), 5)

        # Original pairing / selection logic for cone_positions
        cone_positions = []
        for i in range(0, len(contour_centers)):
            p = 0
            for j in range(0, len(contour_centers)):
                if (
                    i != j
                    and -self.afvig < (contour_centers[i][0] - contour_centers[j][0]) < self.afvig
                    and contour_centers[i][1] < contour_centers[j][1]
                ):
                    # øverste contour gemmes i cone_positions
                    cone_positions.append(contour_centers[i])

                # If alone horizontally beyond afvig: mark + keep
                if (
                    i != j
                    and (-self.afvig > (contour_centers[i][0] - contour_centers[j][0])
                         or self.afvig < (contour_centers[i][0] - contour_centers[j][0]))
                ):
                    p = p + 1
                    if p >= len(contour_centers) - 1:
                        cv2.circle(color_array, (contour_centers[i][0], contour_centers[i][1]), 4, (255, 255, 255), -1)
                        # contour gemmes hvis den står alene
                        cone_positions.append(contour_centers[i])

        return color_array, cone_positions


# -------------------------
# Visualization (final overlays + windows)
# -------------------------
class Visualizer:
    @staticmethod
    def annotate_cones(color_array, depth_frame, cone_positions, intrinsics):
        """
        Intrinsics-based angle computation:
          Angle_rad = atan2( (u - ppx)/fx , 1 )
        Everything else remains the same.
        """
        fx = intrinsics.fx
        ppx = intrinsics.ppx

        for i in range(0, len(cone_positions)):
            u = cone_positions[i][0]  # pixel x
            Angle_rad = math.atan2((u - ppx) / fx, 1.0)

            cv2.circle(color_array, (cone_positions[i][0], cone_positions[i][1]), 4, (255, 255, 255), -1)
            cone_distance = depth_frame.get_distance(int(cone_positions[i][0]), int(cone_positions[i][1]))
            x_coord = abs(math.sin(Angle_rad) * cone_distance)

            cv2.putText(
                color_array,
                f" a:({round(math.degrees(Angle_rad), 2)}, coo:{round(x_coord, 2)},d: {round(cone_distance,2)},c: {cone_positions[i][2]}",
                (cone_positions[i][0], cone_positions[i][1]),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 0, 255)
            )

    @staticmethod
    def show(color_array, depth_img):
        cv.imshow("thresh", color_array)
        cv.imshow("depth", depth_img)


# -------------------------
# Orchestrator
# -------------------------
class Main:
    def __init__(self):
        # Same global values kept via classes
        self.cam = RealSenseCam(res_x=1280, res_y=720, fps=30, fov_x=87)
        self.perception = ConePerception(res_x=1280, fov_x=87, afvig=25)
        self.vis = Visualizer()

    def run(self):
        try:
            while True:
                color_array, depth_frame, depth_img = self.cam.get_processed_frames()
                color_array, cone_positions = self.perception.process_color(color_array)
                # Annotate using accurate intrinsics-based angle
                self.vis.annotate_cones(
                    color_array,
                    depth_frame,
                    cone_positions,
                    self.cam.color_intr  # <— pass intrinsics
                )
                self.vis.show(color_array, depth_img)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.stop()

    def stop(self):
        self.cam.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    Main().run()
