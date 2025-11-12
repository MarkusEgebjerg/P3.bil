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
max_pair_distance = 1.5  # maximum 3D distance (meters) to consider cones as a pair

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


def calculate_3d_distance(pos1, pos2):
    """Calculate 3D Euclidean distance between two 3D positions"""
    return math.sqrt(
        (pos1[0] - pos2[0])**2 +
        (pos1[1] - pos2[1])**2 +
        (pos1[2] - pos2[2])**2
    )


def calculate_midpoint_3d(pos1, pos2):
    """Calculate 3D midpoint between two positions"""
    return (
        (pos1[0] + pos2[0]) / 2,
        (pos1[1] + pos2[1]) / 2,
        (pos1[2] + pos2[2]) / 2
    )


def project_point_to_pixel(X, Y, Z):
    """Convert 3D camera coordinates back to pixel coordinates"""
    u = int(X * fx / Z + ppx)
    v = int(Y * fy / Z + ppy)
    return u, v


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
    blue_cones = []
    yellow_cones = []

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

    # Group nearby contours and pick the topmost one from each group
    used = set()
    cone_positions = []

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

    # Process each cone and get 3D positions
    for u, v, color_label in cone_positions:
        # Get robust depth measurement
        depth = get_robust_depth(depth_frame, u, v, window_size=5)

        if depth <= 0 or depth > 10:  # Invalid or too far
            cv2.putText(color_array, f"Invalid - {color_label}",
                        (u - 40, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            continue

        # Deproject to 3D camera coordinates
        X, Y, Z = deproject_pixel_to_point(u, v, depth)

        # Store cone data: (u, v, X, Y, Z, color)
        cone_data = {
            'pixel': (u, v),
            'position_3d': (X, Y, Z),
            'color': color_label
        }

        if color_label == "Blue":
            blue_cones.append(cone_data)
        else:
            yellow_cones.append(cone_data)

        # Draw marker on cone
        cv2.circle(color_array, (u, v), 5, (255, 255, 255), -1)

    # Pair blue and yellow cones based on 3D proximity
    cone_pairs = []
    used_blue = set()
    used_yellow = set()

    for y_idx, yellow in enumerate(yellow_cones):
        if y_idx in used_yellow:
            continue

        best_match = None
        min_distance = float('inf')

        # Find closest blue cone
        for b_idx, blue in enumerate(blue_cones):
            if b_idx in used_blue:
                continue

            dist = calculate_3d_distance(yellow['position_3d'], blue['position_3d'])

            if dist < min_distance and dist < max_pair_distance:
                min_distance = dist
                best_match = b_idx

        # If we found a match, create a pair
        if best_match is not None:
            cone_pairs.append({
                'blue': blue_cones[best_match],
                'yellow': yellow,
                'distance': min_distance
            })
            used_blue.add(best_match)
            used_yellow.add(y_idx)

    # Draw pairs, distances, and midpoints
    for pair_idx, pair in enumerate(cone_pairs):
        blue = pair['blue']
        yellow = pair['yellow']
        distance = pair['distance']

        blue_pixel = blue['pixel']
        yellow_pixel = yellow['pixel']

        # Draw line connecting the pair
        cv2.line(color_array, blue_pixel, yellow_pixel, (255, 0, 255), 2)

        # Calculate and draw midpoint
        midpoint_3d = calculate_midpoint_3d(blue['position_3d'], yellow['position_3d'])
        mid_u, mid_v = project_point_to_pixel(*midpoint_3d)

        # Draw midpoint
        cv2.circle(color_array, (mid_u, mid_v), 7, (0, 255, 255), -1)
        cv2.circle(color_array, (mid_u, mid_v), 9, (255, 255, 255), 2)

        # Display pair information at midpoint
        mid_X, mid_Y, mid_Z = midpoint_3d
        mid_angle = math.atan2(mid_X, mid_Z) * 180 / math.pi
        mid_distance = math.sqrt(mid_X**2 + mid_Y**2 + mid_Z**2)

        # Text at midpoint
        text1 = f"Pair {pair_idx + 1}: {distance:.2f}m apart"
        text2 = f"Mid: {mid_distance:.2f}m, {mid_angle:+.1f}deg"

        cv2.putText(color_array, text1, (mid_u - 80, mid_v - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(color_array, text2, (mid_u - 80, mid_v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Display info at each cone
        b_X, b_Y, b_Z = blue['position_3d']
        b_angle = math.atan2(b_X, b_Z) * 180 / math.pi
        b_dist = math.sqrt(b_X**2 + b_Y**2 + b_Z**2)

        y_X, y_Y, y_Z = yellow['position_3d']
        y_angle = math.atan2(y_X, y_Z) * 180 / math.pi
        y_dist = math.sqrt(y_X**2 + y_Y**2 + y_Z**2)

        cv2.putText(color_array, f"B: {b_dist:.2f}m {b_angle:+.1f}deg",
                    (blue_pixel[0] - 80, blue_pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(color_array, f"Y: {y_dist:.2f}m {y_angle:+.1f}deg",
                    (yellow_pixel[0] - 80, yellow_pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # Display unpaired cones
    for b_idx, blue in enumerate(blue_cones):
        if b_idx not in used_blue:
            u, v = blue['pixel']
            X, Y, Z = blue['position_3d']
            angle = math.atan2(X, Z) * 180 / math.pi
            dist = math.sqrt(X**2 + Y**2 + Z**2)
            cv2.putText(color_array, f"B(unpaired): {dist:.2f}m {angle:+.1f}deg",
                        (u - 80, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

    for y_idx, yellow in enumerate(yellow_cones):
        if y_idx not in used_yellow:
            u, v = yellow['pixel']
            X, Y, Z = yellow['position_3d']
            angle = math.atan2(X, Z) * 180 / math.pi
            dist = math.sqrt(X**2 + Y**2 + Z**2)
            cv2.putText(color_array, f"Y(unpaired): {dist:.2f}m {angle:+.1f}deg",
                        (u - 80, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)

    cv.imshow("Detection", color_array)
    cv.imshow("Depth", depth_img)

    if cv.waitKey(200) & 0xFF == ord('q'):
        break

pipe.stop()
cv.destroyAllWindows()