import cv2
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import math

# -------- Configuration --------
res_x = 640
res_y = 480
fps = 30
afvig = 25  # horizontal tolerance in pixels

lowerYellow = np.array([20, 110, 90])
upperYellow = np.array([33, 255, 255])

lowerBlue = np.array([105, 120, 60])
upperBlue = np.array([135, 255, 255])

kernel = np.ones((2, 3), np.uint8)

# -------- RealSense Setup --------
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, res_x, res_y, rs.format.bgr8, fps)
cfg.enable_stream(rs.stream.depth, res_x, res_y, rs.format.z16, fps)

stream = pipe.start(cfg)
depth_sensor = stream.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Align depth to color
align = rs.align(rs.stream.color)

# Get camera intrinsics (CRITICAL for accurate measurements)
color_profile = stream.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()
fx = intrinsics.fx  # focal length x
fy = intrinsics.fy  # focal length y
ppx = intrinsics.ppx  # principal point x (optical center)
ppy = intrinsics.ppy  # principal point y (optical center)

# Setup depth filters (configure BEFORE processing)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_fill = rs.hole_filling_filter()

spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
spatial.set_option(rs.option.holes_fill, 0)
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal.set_option(rs.option.filter_smooth_delta, 20)


def Masking(mask):
    """Apply morphological operations to clean up the mask"""
    mask_Open = cv2.erode(mask, kernel, iterations=3)
    mask_Close = cv2.dilate(mask_Open, kernel, iterations=10)
    return mask_Close


def get_robust_depth(depth_frame, u, v, window_size=5):
    """Get robust depth measurement by averaging nearby valid pixels"""
    half_w = window_size // 2
    depths = []

    for dy in range(-half_w, half_w + 1):
        for dx in range(-half_w, half_w + 1):
            px = int(u + dx)
            py = int(v + dy)
            if 0 <= px < res_x and 0 <= py < res_y:
                d = depth_frame.get_distance(px, py)
                if d > 0:
                    depths.append(d)

    if not depths:
        return 0

    # Use median to reject outliers
    return np.median(depths)


def deproject_pixel_to_point(u, v, depth):
    """Convert pixel coordinates and depth to 3D camera coordinates"""
    X = (u - ppx) / fx * depth
    Y = (v - ppy) / fy * depth
    Z = depth
    return X, Y, Z


while True:
    frameset = pipe.wait_for_frames()
    frames = align.process(frameset)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    # Apply filters in correct order (options already set)
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_fill.process(depth_frame)
    depth_frame = depth_frame.as_depth_frame()

    # Convert to numpy arrays
    depth_array = np.asanyarray(depth_frame.get_data())
    depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=0.06), cv2.COLORMAP_JET)

    color_array = np.asanyarray(color_frame.get_data())
    frame_HSV = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)

    # Threshold for yellow and blue
    frame_threshold_Y = cv2.inRange(frame_HSV, lowerYellow, upperYellow)
    frame_threshold_B = cv2.inRange(frame_HSV, lowerBlue, upperBlue)

    Masking_Clean_Y = Masking(frame_threshold_Y)
    Masking_Clean_B = Masking(frame_threshold_B)

    contours_b, _ = cv2.findContours(Masking_Clean_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_y, _ = cv2.findContours(Masking_Clean_Y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_centers = []
    cone_positions = []

    # Find contour centers
    for i in range(2):
        contours_list = contours_b if i == 0 else contours_y

        for contour in contours_list:
            area = cv2.contourArea(contour)
            if area < 60:
                continue

            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            M = cv2.moments(contour)
            if M['m00'] == 0:  # Avoid division by zero
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            label = "Blue" if i == 0 else "Yellow"
            contour_centers.append((cx, cy, label))

            # Draw contour
            if len(approx) > 3:
                cv2.drawContours(color_array, [approx], 0, (255, 0, 0), 5)

    # FIXED PAIRING LOGIC
    # Group nearby contours and pick the topmost one from each group
    used = set()

    for i in range(len(contour_centers)):
        if i in used:
            continue

        xi, yi, _ = contour_centers[i]

        # Find all centers within horizontal tolerance
        neighbors = []
        for j in range(len(contour_centers)):
            if j != i and abs(contour_centers[j][0] - xi) < afvig:
                neighbors.append(j)

        if neighbors:
            # Multiple contours nearby - pick the topmost (smallest y)
            group = [i] + neighbors
            topmost_idx = min(group, key=lambda k: contour_centers[k][1])

            if topmost_idx not in used:
                cone_positions.append(contour_centers[topmost_idx])

            # Mark all in group as used
            used.update(group)
        else:
            # Single contour standing alone
            cone_positions.append(contour_centers[i])
            used.add(i)

    # Calculate and display accurate measurements
    for i in range(len(cone_positions)):
        u, v, color_label = cone_positions[i]

        # Get robust depth measurement
        depth = get_robust_depth(depth_frame, u, v, window_size=5)

        if depth <= 0 or depth > 10:  # Invalid or too far
            cv2.putText(color_array, f"Invalid depth - {color_label}",
                        (u - 50, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            continue

        # Deproject to 3D camera coordinates
        X, Y, Z = deproject_pixel_to_point(u, v, depth)

        # Calculate metrics
        angle_rad = math.atan2(X, Z)  # Horizontal angle from camera optical axis
        angle_deg = angle_rad * 180 / math.pi  # Convert to degrees

        lateral_offset = abs(X)  # Horizontal distance from center axis (meters)
        direct_distance = math.sqrt(X * X + Y * Y + Z * Z)  # True 3D distance (meters)

        # Draw marker on cone
        cv2.circle(color_array, (u, v), 5, (255, 255, 255), -1)

        # Display measurements
        text = f"a:{angle_deg:+.1f}deg off:{lateral_offset:.2f}m d:{direct_distance:.2f}m"
        cv2.putText(color_array, text, (u - 80, v - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Display color
        cv2.putText(color_array, color_label, (u - 30, v + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    cv.imshow("Detection", color_array)
    cv.imshow("Depth", depth_img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

pipe.stop()
cv.destroyAllWindows()