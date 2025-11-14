import cv2
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import math


class Camera:
    def __init__(self, res_x=1280, res_y=720, fps=30, fov_x=87):
        # Stream configuration
        self.res_x = res_x
        self.res_y = res_y
        self.fps = fps
        self.fov_x = fov_x

        # RealSense pipeline + stream setup
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, self.res_x, self.res_y, rs.format.bgr8, self.fps)
        self.cfg.enable_stream(rs.stream.depth, self.res_x, self.res_y, rs.format.z16, self.fps)

        # Start streaming
        self.stream = self.pipe.start(self.cfg)

        # Depth scale (meters per unit)
        depth_sensor = self.stream.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Align depth to color so pixel (x,y) refers to same place in both frames
        self.align = rs.align(rs.stream.color)

        # Pixels per degree (simple linear model used later for angle computation)
        self.PxD = self.res_x / self.fov_x

    def get_frames(self):
        # Grab a synchronized frameset, align depth to color, and return both frames.
        frameset = self.pipe.wait_for_frames()
        frames = self.align.process(frameset)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return color_frame, depth_frame

    def filter_depth(self, depth_frame):
        # Apply spatial filtering for smoothing/hole reduction.
        spatial = rs.spatial_filter()

        # Initial pass
        depth_frame = spatial.process(depth_frame)

        # Configure smoothing parameters
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 1)
        spatial.set_option(rs.option.filter_smooth_delta, 50)
        depth_frame = spatial.process(depth_frame)

        # Hole filling (0 = off; consider >0 for more robustness)
        spatial.set_option(rs.option.holes_fill, 1)
        depth_frame = spatial.process(depth_frame)

        # Ensure depth frame type
        depth_frame = depth_frame.as_depth_frame()
        return depth_frame

    def stop(self):
        self.pipe.stop()


class ConeDetector:
    def __init__(self):
        # HSV thresholds for yellow and blue cones
        self.lowerYellow = np.array([20, 110, 90])
        self.upperYellow = np.array([33, 255, 255])
        self.lowerBlue   = np.array([105, 120, 60])
        self.upperBlue   = np.array([135, 255, 255])

        # Small morphology kernel to reduce noise and close gaps
        self.kernel = np.ones((2, 3), dtype=np.uint8)

    def Masking(self, mask):
        # Clean the binary mask using erosion then dilation.
        mask_Open  = cv2.erode(mask, self.kernel, iterations=3)
        mask_Close = cv2.dilate(mask_Open, self.kernel, iterations=10)
        return mask_Close

    def find_contours(self, color_frame):
        # Convert color frame → HSV → threshold by color → clean masks → find contours.
        color_array = np.asanyarray(color_frame.get_data())
        frame_HSV = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)

        # Color thresholding
        frame_threshold_Y = cv2.inRange(frame_HSV, self.lowerYellow, self.upperYellow)
        frame_threshold_B = cv2.inRange(frame_HSV, self.lowerBlue,  self.upperBlue)

        # Morphology cleanup
        Masking_Clean_Y = self.Masking(frame_threshold_Y)
        Masking_Clean_B = self.Masking(frame_threshold_B)

        # Contour extraction
        contours_b, _ = cv2.findContours(Masking_Clean_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_y, _ = cv2.findContours(Masking_Clean_Y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return color_array, contours_y, contours_b


class ConeCenterPoints:
    def __init__(self, afvig=25, PxD=1280/87, fov_x=87):
        # Horizontal tolerance in pixels to consider two centers "paired"
        self.afvig = afvig
        # Pixels-per-degree and FOV used for angle approximation
        self.PxD = PxD
        self.fov_x = fov_x

    def build_centers_and_draw(self, contours_y, contours_b, color_array):
        # Compute centers (moments) for all contours, label color, draw polygons for visualization.
        contour_centers = []

        # i==0 → blue list; i==1 → yellow list
        for i in range(2):
            lst = contours_b if i == 0 else contours_y
            for contour in lst:
                # Skip very small blobs
                area = cv2.contourArea(contour)
                if area < 60:
                    continue

                # Approximate polygon
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Center from moments
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                label = "Blue" if i == 0 else "Yellow"
                contour_centers.append((cx, cy, label))

                # Visualize contour
                if len(approx) > 3:
                    cv2.drawContours(color_array, [approx], 0, (255, 0, 0), 5)

        return contour_centers






    def select_cone_positions(self, contour_centers, color_array):
        # Pair-selection logic:
        # - If two centers are horizontally close (±afvig), keep the one higher up (smaller y).
        # - If a center is "alone" horizontally (no others within ±afvig), keep it anyway.

        cone_positions = []
        n = len(contour_centers)

        for i in range(n):
            p = 0
            for j in range(n):
                if i == j:
                    continue

                dx = contour_centers[i][0] - contour_centers[j][0]

                # Near-horizontal match: keep upper one
                if -self.afvig < dx < self.afvig and contour_centers[i][1] < contour_centers[j][1]:
                    cone_positions.append(contour_centers[i])

                # Far apart horizontally: count as "alone"
                if (-self.afvig > dx) or (self.afvig < dx):
                    p += 1
                    # If it's far from every other center, keep it and mark
                    if p >= n - 1:
                        cv2.circle(color_array, (contour_centers[i][0], contour_centers[i][1]),
                                   4, (255, 255, 255), -1)
                        cone_positions.append(contour_centers[i])

        return cone_positions

    def annotate(self, color_array, depth_frame, cone_positions):
        # Draw selection dot and text
        for (cx, cy, label) in cone_positions:
            # Angle formula
            Angle = (cx / self.PxD - self.fov_x // 2) * math.pi / 180.0

            # Draw center
            cv2.circle(color_array, (cx, cy), 4, (255, 255, 255), -1)

            # Distance at that pixel (meters)
            cone_distance = depth_frame.get_distance(int(cx), int(cy))

            # Lateral coordinate assuming distance along ray at Angle
            x_coord = abs(np.sin(Angle) * cone_distance)

            # On-screen text with values
            cv2.putText(
                color_array,
                f"a:{round(Angle*(180/math.pi),2)}, coo:{round(x_coord,2)}, d:{round(cone_distance,2)}, c:{label}",
                (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 0, 255)
            )
#class MidWayPoints:


class MAIN:
    def __init__(self):
        # Camera and helpers
        self.cam = Camera(res_x=1280, res_y=720, fps=30, fov_x=87)
        self.detector = ConeDetector()
        self.center_ops = ConeCenterPoints(afvig=25, PxD=self.cam.PxD, fov_x=self.cam.fov_x)

    def run(self):
        try:
            while True:
                # Acquire frames (aligned depth)
                color_frame, depth_frame_raw = self.cam.get_frames()

                # Filtered depth for smoother/cleaner readings
                depth_frame = self.cam.filter_depth(depth_frame_raw)

                # Depth visualization (colored) for debugging
                depth_array = np.asanyarray(depth_frame.get_data())
                depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=0.06), cv2.COLORMAP_JET)

                # Color segmentation + contour extraction
                color_array, contours_y, contours_b = self.detector.find_contours(color_frame)

                # Compute centers, draw contours, and select cone positions
                contour_centers = self.center_ops.build_centers_and_draw(contours_y, contours_b, color_array)
                cone_positions = self.center_ops.select_cone_positions(contour_centers, color_array)

                # Add angle/coord/depth annotations
                self.center_ops.annotate(color_array, depth_frame, cone_positions)

                # Show result windows
                cv.imshow("thresh", color_array)
                cv.imshow("depth", depth_img)

                # Quit on 'q'
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Always release resources
            self.cam.stop()
            cv.destroyAllWindows()


if __name__ == "__main__":
    MAIN().run()
