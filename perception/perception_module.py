import cv2
import numpy as np
import pyrealsense2 as rs
import logging
import time

# Import config
try:
    from config import (
        CAMERA_CONFIG,
        COLOR_THRESHOLDS,
        PERCEPTION_CONFIG,
        MORPHOLOGY_CONFIG,
        SPATIAL_FILTER
    )
except ImportError:
    # Fallback to default values if config.py not found
    CAMERA_CONFIG = {'resolution_x': 1280, 'resolution_y': 720, 'fps': 30,
                     'crop_top_ratio': 0.25, 'crop_bottom_ratio': 0.75}
    COLOR_THRESHOLDS = {'yellow_lower': [22, 110, 120], 'yellow_upper': [33, 255, 255],
                        'blue_lower': [100, 120, 60], 'blue_upper': [135, 255, 255]}
    PERCEPTION_CONFIG = {'min_contour_area': 30, 'neighbor_distance': 25, 'max_depth': 6.0,
                         'z_smoothing_window': 5, 'depth_offset': 180}
    MORPHOLOGY_CONFIG = {'kernel_size': (2, 3), 'erosion_iterations': 2, 'dilation_iterations': 10}
    SPATIAL_FILTER = {'magnitude': 5, 'smooth_alpha': 0.5, 'smooth_delta': 50, 'holes_fill': 5}

logger = logging.getLogger(__name__)


class PerceptionModule:
    def __init__(self):
        # --- RealSense Setup ---
        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        # Use config values
        self.res_x = CAMERA_CONFIG['resolution_x']
        self.res_y = CAMERA_CONFIG['resolution_y']
        self.fps = CAMERA_CONFIG['fps']

        self.cfg.enable_stream(rs.stream.color, self.res_x, self.res_y, rs.format.bgr8, self.fps)
        self.cfg.enable_stream(rs.stream.depth, self.res_x, self.res_y, rs.format.z16, self.fps)
        self.align = rs.align(rs.stream.color)

        # Start streaming
        logger.info("Starting RealSense camera...")
        try:
            self.stream = self.pipe.start(self.cfg)
        except RuntimeError as e:
            logger.error(f"Failed to start RealSense pipeline: {e}")
            raise

        # --- Warm-up RealSense pipeline ---
        logger.info("Warming up camera...")
        self._warmup_camera()

        self.depth_sensor = self.stream.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.depth_intrin = None

        # --- Color Detection Settings from config ---
        self.lowerYellow = np.array(COLOR_THRESHOLDS['yellow_lower'])
        self.upperYellow = np.array(COLOR_THRESHOLDS['yellow_upper'])
        self.lowerBlue = np.array(COLOR_THRESHOLDS['blue_lower'])
        self.upperBlue = np.array(COLOR_THRESHOLDS['blue_upper'])

        # --- Perception settings from config ---
        self.min_contour_area = PERCEPTION_CONFIG['min_contour_area']
        self.neighbor_distance = PERCEPTION_CONFIG['neighbor_distance']
        self.max_depth = PERCEPTION_CONFIG['max_depth']
        self.depth_offset = PERCEPTION_CONFIG['depth_offset']

        # Morphology settings
        self.kernel = np.ones(MORPHOLOGY_CONFIG['kernel_size'])
        self.erosion_iter = MORPHOLOGY_CONFIG['erosion_iterations']
        self.dilation_iter = MORPHOLOGY_CONFIG['dilation_iterations']

        # Image cropping
        self.crop_top = CAMERA_CONFIG['crop_top_ratio']
        self.crop_bottom = CAMERA_CONFIG['crop_bottom_ratio']

        # Spatial filter - FIXED: configure before use
        self.spatial = rs.spatial_filter()
        self._configure_spatial_filter()

        # Z-smoothing
        self.z_window_size = PERCEPTION_CONFIG['z_smoothing_window']
        self.z_histories = {}

        # Health tracking
        self.frame_count = 0
        self.error_count = 0
        self.last_valid_cones = []

        logger.info("Perception module initialized successfully")

    def _warmup_camera(self):
        """Properly warm up the camera and clear buffers"""
        warmup_frames = 30
        for i in range(warmup_frames):
            try:
                frameset = self.pipe.wait_for_frames(timeout_ms=1000)
                # Explicitly release warmup frames
                frameset.release()
            except RuntimeError as e:
                logger.debug(f"Warmup frame {i} dropped (expected): {e}")

        # Additional buffer clearing
        time.sleep(0.5)
        try:
            while self.pipe.poll_for_frames():
                pass
        except:
            pass

        logger.info("Camera warm-up complete")

    def _configure_spatial_filter(self):
        """Configure spatial filter with all options BEFORE processing"""
        try:
            self.spatial.set_option(rs.option.filter_magnitude, SPATIAL_FILTER['magnitude'])
            self.spatial.set_option(rs.option.filter_smooth_alpha, SPATIAL_FILTER['smooth_alpha'])
            self.spatial.set_option(rs.option.filter_smooth_delta, SPATIAL_FILTER['smooth_delta'])
            self.spatial.set_option(rs.option.holes_fill, SPATIAL_FILTER['holes_fill'])
            logger.info("Spatial filter configured successfully")
        except Exception as e:
            logger.warning(f"Could not configure spatial filter: {e}")

    def get_frame(self):
        """Get aligned frames with proper error handling"""
        try:
            frameset = self.pipe.wait_for_frames(timeout_ms=5000)
            frames = self.align.process(frameset)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Validate frames
            if not depth_frame or not color_frame:
                logger.warning("Invalid frame received - missing depth or color")
                return None, None

            self.frame_count += 1
            return depth_frame, color_frame

        except RuntimeError as e:
            self.error_count += 1
            logger.error(f"Frame capture error: {e}")
            return None, None

    def update_intrinsics(self, depth_frame):
        if self.depth_intrin is None:
            self.depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        return self.depth_intrin

    def spatial_filter(self, depth_frame):
        """Apply spatial filtering - FIXED: set options before process"""
        try:
            filtered_frame = self.spatial.process(depth_frame)
            return filtered_frame.as_depth_frame()
        except Exception as e:
            logger.warning(f"Spatial filter failed: {e}, returning unfiltered frame")
            return depth_frame

    def color_space_conversion(self, color_frame):
        """Convert to HSV and crop using config values"""
        color_array = np.asanyarray(color_frame.get_data())
        res_y, res_x = color_array.shape[:2]

        # Use config crop ratios
        crop_start = int(res_y * self.crop_top)
        crop_end = int(res_y * self.crop_bottom)
        color_array = color_array[crop_start:crop_end, 0:res_x]

        frame_HSV = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)
        return frame_HSV, color_array

    def mask_clean(self, mask):
        """Clean mask using config morphology settings"""
        mask = cv2.erode(mask, self.kernel, iterations=self.erosion_iter)
        mask = cv2.dilate(mask, self.kernel, iterations=self.dilation_iter)
        return mask

    def color_detector(self, frame_HSV):
        """Detect colors using config thresholds"""
        clean_mask_y = self.mask_clean(cv2.inRange(frame_HSV, self.lowerYellow, self.upperYellow))
        clean_mask_b = self.mask_clean(cv2.inRange(frame_HSV, self.lowerBlue, self.upperBlue))
        return clean_mask_y, clean_mask_b

    def find_contour(self, clean_mask_y, clean_mask_b):
        contours_b, _ = cv2.findContours(clean_mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_y, _ = cv2.findContours(clean_mask_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_y, contours_b

    def find_centers(self, contours_y, contours_b, img):
        """Find contour centers using bounding box method"""
        self.contour_centers = []

        for i in range(2):
            contour_list = contours_b if i == 0 else contours_y

            for contour in contour_list:
                area = cv2.contourArea(contour)
                if area < self.min_contour_area:
                    continue

                # Bounding box method
                x, y, w, h = cv2.boundingRect(contour)
                cx = int(x + w / 2)
                cy = int(y + h / 2)

                color_label = "Blue" if i == 0 else "Yellow"
                self.current_center = (cx, cy, color_label)
                self.contour_centers.append(self.current_center)

        return self.contour_centers

    def contour_control(self, contour_centers, img):
        """Filter contours to find cone tops using config neighbor distance"""
        self.cone_positions = []

        for i in range(len(contour_centers)):
            p = 0
            for j in range(len(contour_centers)):
                # Use config neighbor_distance
                if i != j and -self.neighbor_distance < (
                        contour_centers[i][0] - contour_centers[j][0]) < self.neighbor_distance and contour_centers[i][
                    1] < contour_centers[j][1]:
                    self.cone_positions.append(contour_centers[i])
                    break

                if i != j and (-self.neighbor_distance > (
                        contour_centers[i][0] - contour_centers[j][0]) or self.neighbor_distance < (
                                       contour_centers[i][0] - contour_centers[j][0])):
                    p += 1
                    if p >= len(contour_centers) - 1:
                        self.cone_positions.append(contour_centers[i])

            if len(contour_centers) == 1:
                self.cone_positions.append(contour_centers[i])

        return self.cone_positions, img

    def smooth_z(self, key, new_z: float) -> float:
        """Smooth Z values using rolling window from config"""
        if new_z <= 0:
            return new_z

        history = self.z_histories.get(key, [])
        history.append(new_z)

        if len(history) > self.z_window_size:
            history.pop(0)

        self.z_histories[key] = history
        return sum(history) / len(history)

    def world_positioning(self, cone_positions, depth_frame, depth_intrin, img):
        """Convert pixel positions to 3D world coordinates"""
        world_cones = []

        for i in range(len(cone_positions)):
            try:
                u = float(cone_positions[i][0])
                v = float(cone_positions[i][1])

                # Use config depth_offset
                depth_m = float(depth_frame.get_distance(int(u), int(v) + self.depth_offset))

                if depth_m <= 0:
                    continue

                X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth_m)
                color = cone_positions[i][2]

                X = round(X, 2)
                Y = round(Y, 2)
                Z = round(Z, 2)

                # Use config max_depth
                if Z < self.max_depth:
                    world_cones.append((X, Z, color, u, v))
            except Exception as e:
                logger.debug(f"Error processing cone position {i}: {e}")
                continue

        if world_cones:
            self.last_valid_cones = world_cones

        return world_cones, img

    def run(self):
        """Main perception pipeline with error handling"""
        try:
            depth_frame, color_frame = self.get_frame()
            if depth_frame is None or color_frame is None:
                logger.debug("Skipping frame - invalid capture")
                return self.last_valid_cones, None

            depth_intrin = self.update_intrinsics(depth_frame)
            # Spatial filter disabled by default due to processing block issues
            # depth_frame = self.spatial_filter(depth_frame)

            frame_HSV, img = self.color_space_conversion(color_frame)
            clean_mask_y, clean_mask_b = self.color_detector(frame_HSV)
            contours_y, contours_b = self.find_contour(clean_mask_y, clean_mask_b)
            contour_centers = self.find_centers(contours_y, contours_b, img)
            cone_positions, img = self.contour_control(contour_centers, img)
            world_pos, img = self.world_positioning(cone_positions, depth_frame, depth_intrin, img)

            return world_pos, img

        except Exception as e:
            self.error_count += 1
            logger.error(f"Perception pipeline error: {e}", exc_info=False)
            return self.last_valid_cones, None

    def shutdown(self):
        """Clean shutdown of camera"""
        logger.info("Shutting down perception module...")
        try:
            self.pipe.stop()
            cv2.destroyAllWindows()
            logger.info(f"Perception stats - Frames: {self.frame_count}, Errors: {self.error_count}")
        except Exception as e:
            logger.error(f"Error during camera shutdown: {e}")