import cv2
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import math

# ---------------- RealSense setup ----------------
pipe = rs.pipeline()
cfg = rs.config()
res_x = 1280
res_y = 720
fps = 30

cfg.enable_stream(rs.stream.color, res_x, res_y, rs.format.bgr8, fps)
cfg.enable_stream(rs.stream.depth, res_x, res_y, rs.format.z16, fps)

stream = pipe.start(cfg)
depth_sensor = stream.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# ---------------- Color thresholds ----------------
lowerYellow = np.array([20, 110, 90])
upperYellow = np.array([33, 255, 255])

lowerBlue = np.array([105, 120, 60])
upperBlue = np.array([135, 255, 255])

kernel = np.ones((2, 3), np.uint8)
afvig = 25

align = rs.align(rs.stream.color)

# Will be filled once we have a valid depth frame
depth_intrin = None

# Depth filters (created once)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()


def Masking(mask):
    mask_Open = cv2.erode(mask, kernel, iterations=3)
    mask_Close = cv2.dilate(mask_Open, kernel, iterations=10)
    return mask_Close


while True:
    frameset = pipe.wait_for_frames()
    frames = align.process(frameset)

    # Get aligned depth + color frames
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    # Get intrinsics once from the (aligned) depth frame
    if depth_intrin is None:
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    # -------- Depth filtering --------
    depth_frame = spatial.process(depth_frame)

    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    depth_frame = spatial.process(depth_frame)

    spatial.set_option(rs.option.holes_fill, 0)
    depth_frame = spatial.process(depth_frame)
    depth_frame = depth_frame.as_depth_frame()

    # For visualization only
    depth_array = np.asanyarray(depth_frame.get_data())
    depth_img = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_array, alpha=0.06),
        cv2.COLORMAP_JET
    )

    color_array = np.asanyarray(color_frame.get_data())
    frame_HSV = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)

    # -------- Color thresholding --------
    frame_threshold_Y = cv2.inRange(frame_HSV, lowerYellow, upperYellow)
    frame_threshold_B = cv2.inRange(frame_HSV, lowerBlue, upperBlue)

    Masking_Clean_Y = Masking(frame_threshold_Y)
    Masking_Clean_B = Masking(frame_threshold_B)

    contours_b, _ = cv2.findContours(Masking_Clean_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_y, _ = cv2.findContours(Masking_Clean_Y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_centers = []
    cone_positions = []

    # -------- Find contour centers and label by color --------
    for i in range(2):
        lst = contours_b if i == 0 else contours_y

        for contour in lst:
            area = cv2.contourArea(contour)
            if area < 60:
                continue

            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue

            cx_c = int(M['m10'] / M['m00'])
            cy_c = int(M['m01'] / M['m00'])

            color_name = "Blue" if i == 0 else "Yellow"
            current_center = (cx_c, cy_c, color_name)
            contour_centers.append(current_center)

            if len(approx) > 3:
                cv2.drawContours(color_array, [approx], 0, (255, 0, 0), 5)

    # -------- Pair top cones / single cones into cone_positions --------
    for i in range(0, len(contour_centers)):
        p = 0
        for j in range(0, len(contour_centers)):
            # Two centers close in x, one above the other
            if i != j and -afvig < (contour_centers[i][0] - contour_centers[j][0]) < afvig \
                    and contour_centers[i][1] < contour_centers[j][1]:
                cone_positions.append(contour_centers[i])  # upper contour saved

            # Single cone (no other within afvig in x)
            if i != j and (
                -afvig > (contour_centers[i][0] - contour_centers[j][0]) or
                afvig < (contour_centers[i][0] - contour_centers[j][0])
            ):
                p += 1
                if p >= len(contour_centers) - 1:
                    cv2.circle(
                        color_array,
                        (contour_centers[i][0], contour_centers[i][1]),
                        4, (255, 255, 255), -1
                    )
                    cone_positions.append(contour_centers[i])

    # -------- Compute distances & angle (Z, ground, 3D) --------
    for i in range(0, len(cone_positions)):
        u = float(cone_positions[i][0])  # pixel x
        v = float(cone_positions[i][1])  # pixel y

        # Depth value (meters) at this pixel (this is along Z axis!)
        depth_m = float(depth_frame.get_distance(int(u), int(v)))
        if depth_m <= 0:
            continue  # invalid depth

        # 3D point in camera coordinates (X, Y, Z) in meters
        X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth_m)

        # Horizontal angle relative to camera optical axis
        angle_rad = math.atan2(X, Z)
        angle_deg = angle_rad * 180.0 / math.pi

        lateral = X          # left/right
        forward_Z = Z        # forward (depth axis, same idea as get_distance)
        ground_dist = math.hypot(X, Z)            # distance in X-Z plane
        direct_dist = math.sqrt(X*X + Y*Y + Z*Z)  # full 3D distance

        cv2.circle(color_array, (int(u), int(v)), 4, (255, 255, 255), -1)
        text = (
            f"a:{angle_deg:.2f}, "
            f"x:{lateral:.2f}m, "
            f"Z:{forward_Z:.2f}m, "   # depth axis
            #f"Rg:{ground_dist:.2f}m, "  # ground distance (what you probably expect)
            f"R3:{direct_dist:.2f}m, "
            f"c:{cone_positions[i][2]}"
        )
        cv2.putText(
            color_array, text, (int(u)-250, int(v)),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1
        )

    cv.imshow("thresh", color_array)
    #cv.imshow("depth", depth_img)
    # cv.imshow("clean mask y", Masking_Clean_Y)
    # cv.imshow("clean mask b", Masking_Clean_B)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

pipe.stop()
cv.destroyAllWindows()
