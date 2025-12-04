import cv2
import numpy as np
import pyrealsense2 as rs
from logic.logic_module import LogicModule

class PerceptionModule():
    def __init__(self):
        self.logic = LogicModule()
        # --- RealSense Setup ---
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



        self.pipe = rs.pipeline()



        # --- Color Detection Settings ---
        self.lowerYellow = np.array([20, 110, 90])
        self.upperYellow = np.array([33, 255, 255])
        self.lowerBlue = np.array([105, 120, 60])
        self.upperBlue = np.array([135, 255, 255])

        self.kernel = np.ones((2, 3))
        self.afvig = 25

        self.spatial = rs.spatial_filter()


    def get_frame(self):
        frameset = self.pipe.wait_for_frames()
        frames = self.align.process(frameset)
        return frames.get_depth_frame(), frames.get_color_frame()

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

        return depth_frame.as_depth_frame()

    def color_space_conversion (self, color_frame):
        img = np.asanyarray(color_frame.get_data())
        res_y, res_x = img.shape[:2]
        img = img[res_y//2:res_y, 0:res_x]
        frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return frame_HSV, img

    def mask_clean(self, mask):
        mask = cv2.erode(mask, self.kernel, iterations=2)
        mask = cv2.dilate(mask, self.kernel, iterations=10)
        return mask

    def color_detector(self, frame_HSV):
        clean_mask_y = self.mask_clean(cv2.inRange(frame_HSV, self.lowerYellow, self.upperYellow))
        clean_mask_b = self.mask_clean(cv2.inRange(frame_HSV, self.lowerBlue, self.upperBlue))
        return clean_mask_y, clean_mask_b


    def find_contour(self,clean_mask_y, clean_mask_b):
        contours_b, _ = cv2.findContours(clean_mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_y, _ = cv2.findContours(clean_mask_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_y, contours_b


    def find_centers(self, contours_y, contours_b,img): #finds contour center and draws contour
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
                if M["m00"] == 0:
                    continue

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
                    cv2.drawContours(img, [approx], 0, (255, 0, 0), 5)
        return self.contour_centers

    def contour_control(self, contour_centers, img):
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
                        #cv2.circle(img, (contour_centers[i][0], contour_centers[i][1]), 4, (255, 255, 255), -1)
                        self.cone_positions.append(contour_centers[i])  # contour gemmes hvis den står alane
                if len(contour_centers) == 1:
                    self.cone_positions.append(contour_centers[i])  # contour gemmes hvis den står alane
        return self.cone_positions, img

    def world_positioning(self, cone_positions, depth_frame, depth_intrin, img):
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
                cv2.circle(img, (cone_positions[i][0], cone_positions[i][1]), 4, (255, 255, 255), -1)
                cv2.putText(img, f" c: {(cone_positions[i][2])} coo: {[X, Z]}", (int(u), int(v)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return  world_cones, img

    def run(self):
        depth_frame, color_frame = self.get_frame()
        if depth_frame is None:
            return [], None

        depth_intrin = self.update_intrinsics(depth_frame)
        depth_frame = self.spatial_filter(depth_frame)
        frame_HSV, img = self.color_space_conversion(color_frame)
        clean_mask_y, clean_mask_b = self.color_detector(frame_HSV)
        contours_y, contours_b = self.find_contour(clean_mask_y, clean_mask_b)
        contour_centers = self.find_centers(contours_y, contours_b, img)
        cone_positions, img = self.contour_control(contour_centers, img)

        world_pos, img = self.world_positioning(cone_positions, depth_frame, depth_intrin, img)
        blue_cones, yellow_cones = self.logic.cone_sorting(world_pos)
        midpoints = self.logic.cone_midpoints(blue_cones, yellow_cones, img)

        target = self.logic.Interpolation(midpoints)
        if target is not None:
            steering = self.logic.steering_angle(target)
            # for now: just print it so you see it works
            print(f"Target: {target}, steering angle: {steering:.3f} deg")
        else:
            print("No target found (no midpoints)")

        #cv2.imshow("thresh", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return [], None

        return world_pos, img

    def shutdown(self):
        self.pipe.stop()
        cv2.destroyAllWindows()