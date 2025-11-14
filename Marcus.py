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

        # Kept for compatibility – not used for angle now
        self.PxD = self.res_x / self.fov_x

        # Intrinsics for accurate angle computation
        color_profile = self.stream.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intr = color_profile.get_intrinsics()  # fx, fy, ppx, ppy

        # --- Depth filters (tuned) ---
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 1)
        self.spatial.set_option(rs.option.filter_smooth_delta, 50)
        self.spatial.set_option(rs.option.holes_fill, 3)   # stronger hole filling than 0

        self.temporal = rs.temporal_filter()               # actually used now

        # Optional disparity domain (often improves hole filling):
        self.to_disp = rs.disparity_transform(True)
        self.to_depth = rs.disparity_transform(False)

    def get_processed_frames(self):
        """Get aligned frames + apply the same filtering/processing as in the original code, but with temporal + hole fill."""
        frameset = self.pipe.wait_for_frames()
        frames = self.align.process(frameset)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Process depth in disparity domain → spatial → temporal → back to depth
        depth_frame = self.to_disp.process(depth_frame)
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.to_depth.process(depth_frame)

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
        self.lowerYellow = np.array([20, 110, 90])
        self.upperYellow = np.array([33, 255, 255])

        self.lowerBlue = np.array([105, 120, 60])
        self.upperBlue = np.array([135, 255, 255])

        self.kernel = np.ones((2, 3))
        self.afvig = afvig

        self.PxD = res_x / fov_x
        self.fov_x = fov_x

    def Masking(self, mask):
        mask_Open = cv2.erode(mask, self.kernel, iterations=3)
        mask_Close = cv2.dilate(mask_Open, self.kernel, iterations=10)
        return mask_Close

    def process_color(self, color_array):
        frame_HSV = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)
        frame_threshold_Y = cv2.inRange(frame_HSV, self.lowerYellow, self.upperYellow)
        frame_threshold_B = cv2.inRange(frame_HSV, self.lowerBlue, self.upperBlue)

        Masking_Clean_Y = self.Masking(frame_threshold_Y)
        Masking_Clean_B = self.Masking(frame_threshold_B)

        contours_b, _ = cv2.findContours(Masking_Clean_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_y, _ = cv2.findContours(Masking_Clean_Y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                    and (-self.afvig > (contour_centers[i][0] - contour_centers[j][0])
                         or self.afvig < (contour_centers[i][0] - contour_centers[j][0]))
                ):
                    p = p + 1
                    if p >= len(contour_centers) - 1:
                        cv2.circle(color_array, (contour_centers[i][0], contour_centers[i][1]), 4, (255, 255, 255), -1)
                        cone_positions.append(contour_centers[i])

        return color_array, cone_positions


# -------------------------
# Visualization (final overlays + windows)
# -------------------------
class Visualizer:
    @staticmethod
    def depth_median_at(depth_frame, x, y, win=5, min_valid=6):
        """
        Robust depth: median over a window around (x,y) using only positive (valid) depths.
        Returns NaN if not enough valid samples.
        """
        w = depth_frame.get_width()
        h = depth_frame.get_height()
        # Clamp window to image
        half = win // 2
        x0 = max(0, x - half)
        x1 = min(w - 1, x + half)
        y0 = max(0, y - half)
        y1 = min(h - 1, y + half)

        vals = []
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                d = depth_frame.get_distance(xx, yy)
                if d > 0:  # keep only valid
                    vals.append(d)

        if len(vals) < min_valid:
            return float('nan')
        return float(np.median(vals))

    @staticmethod
    def annotate_cones(color_array, depth_frame, cone_positions, intrinsics, sample_bias_px=4):
        """
        Angle from intrinsics; distance = robust median over NxN window (biased a few pixels downward to avoid edge).
        """
        fx = intrinsics.fx
        ppx = intrinsics.ppx

        for i in range(0, len(cone_positions)):
            u = int(cone_positions[i][0])
            v = int(cone_positions[i][1])

            # Slightly bias downwards into the cone body for more stable depth (optional but helpful)
            v_biased = max(0, min(depth_frame.get_height()-1, v + sample_bias_px))

            # Angle from optical axis (radians)
            Angle_rad = math.atan2((u - ppx) / fx, 1.0)

            # Robust depth (meters)
            cone_distance = Visualizer.depth_median_at(depth_frame, u, v_biased, win=7, min_valid=8)

            # If not enough valid samples, fall back to single-pixel read
            if math.isnan(cone_distance) or cone_distance <= 0:
                cone_distance = depth_frame.get_distance(u, v_biased)

            # Lateral x component relative to camera (m)
            x_coord = abs(math.sin(Angle_rad) * cone_distance)

            # Draw + annotate
            cv2.circle(color_array, (u, v), 4, (255, 255, 255), -1)
            cv2.putText(
                color_array,
                f"a:{round(math.degrees(Angle_rad),2)}  x:{round(x_coord,2)}  d:{round(cone_distance,2)}  c:{cone_positions[i][2]}",
                (u, v),
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
        self.cam = RealSenseCam(res_x=1280, res_y=720, fps=30, fov_x=87)
        self.perception = ConePerception(res_x=1280, fov_x=87, afvig=25)
        self.vis = Visualizer()

    def run(self):
        try:
            while True:
                color_array, depth_frame, depth_img = self.cam.get_processed_frames()
                color_array, cone_positions = self.perception.process_color(color_array)

                self.vis.annotate_cones(
                    color_array,
                    depth_frame,
                    cone_positions,
                    self.cam.color_intr
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
