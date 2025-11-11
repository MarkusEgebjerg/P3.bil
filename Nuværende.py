import cv2
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from cv2.gapi import kernel

pipe = rs.pipeline()
cfg = rs.config()
res_x = 1280
res_y = 720
fps = 30
fov_x = 87

cfg.enable_stream(rs.stream.color, res_x, res_y, rs.format.bgr8, fps)
cfg.enable_stream(rs.stream.depth, res_x, res_y, rs.format.z16, fps)

stream = pipe.start(cfg)
depth_sensor = stream.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

lowerYellow = np.array([20, 110, 90])
upperYellow = np.array([33, 255, 255])

lowerBlue = np.array([105, 120, 60])
upperBlue = np.array([135, 255, 255])
kernel = np.ones((2, 3))
afvig = 25

PxD = res_x / fov_x
align = rs.align(rs.stream.color)


def Masking(mask):
    mask_Open = cv2.erode(mask, kernel, iterations=3)
    mask_Close = cv2.dilate(mask_Open, kernel, iterations=10)
    # mask_Open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # mask_Close = cv2.morphologyEx(mask_Close_1, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask_Close


while True:
    frameset = pipe.wait_for_frames()
    frames = align.process(frameset)

    # Tager dybde- og farve billeddet af samlingen "frames"
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Laver om til numpy array
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()

    depth_frame = spatial.process(depth_frame)

    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    depth_frame = spatial.process(depth_frame)

    spatial.set_option(rs.option.holes_fill, 0)
    depth_frame = spatial.process(depth_frame)
    depth_frame = depth_frame.as_depth_frame()

    # depth_frame3 = temporal.process(depth_frame)
    depth_array = np.asanyarray(depth_frame.get_data())
    depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=0.06), cv2.COLORMAP_JET)

    color_array = np.asanyarray(color_frame.get_data())
    frame_HSV = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)
    frame_threshold_Y = cv2.inRange(frame_HSV, lowerYellow, upperYellow)
    frame_threshold_B = cv2.inRange(frame_HSV, lowerBlue, upperBlue)
    # Mega_Mask = cv2.bitwise_or(frame_threshold_B, frame_threshold_Y)
    Masking_Clean_Y = Masking(frame_threshold_Y)
    Masking_Clean_B = Masking(frame_threshold_B)
    contours_b, _ = cv2.findContours(Masking_Clean_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_y, _ = cv2.findContours(Masking_Clean_Y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_centers = []
    cone_positions = []

    for i in range(2):
        list = contours_b
        if i == 1:
            list = contours_y

        for contour in list:
            # Step 4: Approximate the contour.......
            area = cv2.contourArea(contour)
            if area < 60:
                continue

            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if i == 1:
                current_center = (cx, cy, "Yellow")
                contour_centers.append(current_center)
            elif i == 0:
                current_center = (cx, cy, "Blue")
                contour_centers.append(current_center)

            # Draw the contour
            if len(approx) > 3:
                cv2.drawContours(color_array, [approx], 0, (255, 0, 0), 5)
                # cv2.circle(canny, (cx, cy), 4, (255, 255, 255), -1)

    # Loops through alle contour centers. checks if two of them are less than "Afvig" pixels away horizontally (x).
    # if there are, the center highest up (biggest y) is added to cone_positions.
    for i in range(0, len(contour_centers)):
        p = 0
        for j in range(0, len(contour_centers)):

            if i != j and -afvig < (contour_centers[i][0] - contour_centers[j][0]) < afvig and contour_centers[i][1] < \
                    contour_centers[j][1]:
                cone_positions.append(contour_centers[i])  #øverste contour gemmes i cone_positions

            if i != j and -afvig > (contour_centers[i][0] - contour_centers[j][0]) or afvig < (contour_centers[i][0] - contour_centers[j][0]):
                p = p + 1
                if p >= len(contour_centers) - 1:
                    cv2.circle(color_array, (contour_centers[i]), 4, (255, 255, 255), -1)
                    cone_positions.append(contour_centers[i])    #contour gemmes hvis den står alane


    for i in range(0, len(cone_positions)):
        Angle = cone_positions[i][0] / PxD - 43.5
        cv2.circle(color_array, (cone_positions[i][0],cone_positions[i][1] ), 4, (255, 255, 255), -1)
        cone_distance = depth_frame.get_distance(int(cone_positions[i][0]), int(cone_positions[i][0]))
        cv2.putText(color_array, f"Angle: ({(round(Angle, 2))}, {(round(cone_distance, 2))}, {(cone_positions[i][2])}", (cone_positions[i][0],cone_positions[i][1]),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))


    cv.imshow("thresh", color_array)
    cv.imshow("depth", depth_img)
    # cv.imshow("clean mask y", Masking_Clean_Y)
    # cv.imshow("clean mask b", Masking_Clean_B)
    if cv.waitKey(200) & 0xFF == ord('q'):
        break
pipe.stop()