import cv2
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import math

class Camera:
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
        self.PxD = self.res_x / self.fov_x

    def get_frames(self):
        frameset = self.pipe.wait_for_frames()
        frames = self.align.process(frameset)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return color_frame, depth_frame

    def filter_depth(self, depth_frame):
        spatial = rs.spatial_filter()
        depth_frame = spatial.process(depth_frame)
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 1)
        spatial.set_option(rs.option.filter_smooth_delta, 50)
        depth_frame = spatial.process(depth_frame)

        spatial.set_option(rs.option.holes_fill, 0)
        depth_frame = spatial.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()
        return depth_frame

    def stop(self):
        self.pipe.stop()


class ConeDetector:
    def __init__(self):
        # Your thresholds and kernel
        self.lowerYellow = np.array([20, 110, 90])
        self.upperYellow = np.array([33, 255, 255])
        self.lowerBlue = np.array([105, 120, 60])
        self.upperBlue = np.array([135, 255, 255])
        self.kernel = np.ones((2, 3), dtype=np.uint8)

    def Masking(self, mask):
        mask_Open = cv2.erode(mask, self.kernel, iterations=3)
        mask_Close = cv2.dilate(mask_Open, self.kernel, iterations=10)
        return mask_Close

    def find_contours(self, color_frame):
        color_array = np.asanyarray(color_frame.get_data())
        frame_HSV = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)

        frame_threshold_Y = cv2.inRange(frame_HSV, self.lowerYellow, self.upperYellow)
        frame_threshold_B = cv2.inRange(frame_HSV, self.lowerBlue, self.upperBlue)

        Masking_Clean_Y = self.Masking(frame_threshold_Y)
        Masking_Clean_B = self.Masking(frame_threshold_B)

        contours_b, _ = cv2.findContours(Masking_Clean_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_y, _ = cv2.findContours(Masking_Clean_Y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return color_array, contours_y, contours_b

class ConeCenterPoints:
    def __init__(self, afvig=25, PxD=1280/87, fov_x=87):
        self.afvig = afvig
        self.PxD = PxD
        self.fov_x = fov_x

    def build_centers_and_draw(self, contours_y, contours_b, color_array):
        contour_centers = []

        for i in range(2):
            lst = contours_b if i == 0 else contours_y
            for contour in lst:
                area = cv2.contourArea(contour)
                if area < 60:
                    continue

                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                label = "Blue" if i == 0 else "Yellow"
                contour_centers.append((cx, cy, label))

                if len(approx) > 3:
                    cv2.drawContours(color_array, [approx], 0, (255, 0, 0), 5)

        return contour_centers

    def select_cone_positions(self, contour_centers, color_array):
        cone_positions = []
        n = len(contour_centers)

        for i in range(n):
            p = 0
            for j in range(n):
                if i == j:
                    continue

                dx = contour_centers[i][0] - contour_centers[j][0]
                # “Paired” horizontally within +/- afvig and higher up
                if -self.afvig < dx < self.afvig and contour_centers[i][1] < contour_centers[j][1]:
                    cone_positions.append(contour_centers[i])

                # “Alone” horizontally beyond afvig: keep it
                if (-self.afvig > dx) or (self.afvig < dx):
                    p += 1
                    if p >= n - 1:
                        cv2.circle(color_array, (contour_centers[i][0], contour_centers[i][1]),
                                   4, (255, 255, 255), -1)
                        cone_positions.append(contour_centers[i])

        return cone_positions

    def annotate(self, color_array, depth_frame, cone_positions):
        for (cx, cy, label) in cone_positions:
            # Your original angle formula (kept)
            Angle = (cx / self.PxD - self.fov_x // 2) * math.pi / 180.0

            cv2.circle(color_array, (cx, cy), 4, (255, 255, 255), -1)
            cone_distance = depth_frame.get_distance(int(cx), int(cy))
            x_coord = abs(np.sin(Angle) * cone_distance)

            cv2.putText(
                color_array,
                f"a:{round(Angle*(180/math.pi),2)}, coo:{round(x_coord,2)}, d:{round(cone_distance,2)}, c:{label}",
                (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 0, 255)
            )

class MAIN:
    def __init__(self):
        # Camera and helpers
        self.cam = Camera(res_x=1280, res_y=720, fps=30, fov_x=87)
        self.detector = ConeDetector()
        self.center_ops = ConeCenterPoints(afvig=25, PxD=self.cam.PxD, fov_x=self.cam.fov_x)

    def run(self):
        try:
            while True:
                # Get frames
                color_frame, depth_frame_raw = self.cam.get_frames()
                depth_frame = self.cam.filter_depth(depth_frame_raw)

                # Visual depth image (same as before)
                depth_array = np.asanyarray(depth_frame.get_data())
                depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=0.06), cv2.COLORMAP_JET)

                # Segment + contours
                color_array, contours_y, contours_b = self.detector.find_contours(color_frame)

                # Centers + pairing
                contour_centers = self.center_ops.build_centers_and_draw(contours_y, contours_b, color_array)
                cone_positions = self.center_ops.select_cone_positions(contour_centers, color_array)

                # Annotate (angle/coord/distance text)
                self.center_ops.annotate(color_array, depth_frame, cone_positions)

                cv.imshow("thresh", color_array)
                cv.imshow("depth", depth_img)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cam.stop()
            cv.destroyAllWindows()

if __name__ == "__main__":
    MAIN().run()
