import cv2
import numpy as np
import pyrealsense2 as rs

# ========================= RealSense RGB+Depth Handler (robust) ========================= #
class RealSenseRGBHandler:
    def __init__(self, desired_color=(1920, 1080, 15)):
        self.pipe = rs.pipeline()

        # ---- Enumerate supported color profiles (no streaming yet) ----
        ctx = rs.context()
        if len(ctx.devices) == 0:
            raise RuntimeError("No RealSense device found.")
        dev = ctx.devices[0]

        color_profiles = []  # (w,h,fps,fmt)
        for sensor in dev.query_sensors():
            for p in sensor.get_stream_profiles():
                if not p.is_video_stream_profile():
                    continue
                v = p.as_video_stream_profile()
                st = p.stream_type()
                fmt = p.format()
                trip = (v.width(), v.height(), v.fps(), fmt)
                if st == rs.stream.color and fmt in (rs.format.bgr8, rs.format.rgb8):
                    color_profiles.append(trip)

        if not color_profiles:
            raise RuntimeError("No usable color profiles found (BGR8/RGB8).")

        # Dedup + sort while ignoring fmt (avoid comparing enum objects)
        color_profiles = sorted(set(color_profiles), key=lambda t: (t[0], t[1], t[2]))

        # Prefer FPS close to desired, then 60, 30, 15
        fps_preference = []
        for f in (desired_color[2], 60, 30, 15):
            if f not in fps_preference:
                fps_preference.append(f)

        def closeness_2d(a, b_wh):
            # smaller is better: sum |Δw|+|Δh|
            return abs(a[0] - b_wh[0]) + abs(a[1] - b_wh[1])

        started = False
        chosen_color = None

        for fps in fps_preference:
            cand_colors = [c for c in color_profiles if c[2] == fps]
            if not cand_colors:
                continue

            cand_colors.sort(
                key=lambda c: (closeness_2d(c, (desired_color[0], desired_color[1])), -c[2])
            )

            for c in cand_colors:
                cfg = rs.config()
                try:
                    # Enable color stream (chosen profile)
                    cfg.enable_stream(rs.stream.color, c[0], c[1], c[3], c[2])
                    # Enable depth stream (fixed resolution, same fps – usually valid on D4xx)
                    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, c[2])

                    prof = self.pipe.start(cfg)
                    self.profile = prof
                    self.cfg = cfg
                    chosen_color = c
                    started = True
                    break
                except Exception:
                    try:
                        self.pipe.stop()
                    except Exception:
                        pass
                    continue
            if started:
                break

        if not started:
            # Safe fallback: 640x480 color + depth @30
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.profile = self.pipe.start(cfg)
            chosen_color = (640, 480, 30, rs.format.rgb8)

        self.color_mode = chosen_color  # (w,h,fps,fmt)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Depth scale (raw units -> meters)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        # Color intrinsics (if you need them later for XYZ)
        self.color_intr = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        # RGB vs BGR flag
        self.color_is_rgb = (self.color_mode[3] == rs.format.rgb8)

    def get_frame(self):
        frames = self.pipe.wait_for_frames()
        # Align depth to color frame
        frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None, None

        color = np.asanyarray(color_frame.get_data())  # uint8, RGB or BGR
        if self.color_is_rgb:
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return True, color, depth_frame

    def release(self):
        self.pipe.stop()


# ========================= Cone Detector with Canny + HSV + Depth ========================= #
class ConeDetectorCanny:
    def __init__(self,
                 min_area=400,
                 canny_thresh1=50,
                 canny_thresh2=150,
                 max_distance_m=15.0,
                 depth_scale=0.001):
        # HSV ranges for yellow + blue cones (tweak if needed)
        self.lowerYellow = np.array([18, 110,  90], dtype=np.uint8)
        self.upperYellow = np.array([35, 255, 255], dtype=np.uint8)

        self.lowerBlue   = np.array([100, 120, 60], dtype=np.uint8)
        self.upperBlue   = np.array([140, 255, 255], dtype=np.uint8)

        self.kernel = np.ones((5, 5), np.uint8)

        self.min_area = min_area
        self.canny_t1 = canny_thresh1
        self.canny_t2 = canny_thresh2
        self.max_distance_m = max_distance_m  # ignore cones beyond this distance
        self.depth_scale = depth_scale        # from RealSense handler

    def _clean_mask(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        return mask

    def process(self, frame_bgr, depth_frame):
        """
        Takes a BGR frame + depth_frame, finds yellow & blue cone-like blobs using
        HSV + Canny + contours. Depth is used in TWO ways:

        1) Per-pixel gating: masks are zeroed where depth > max_distance_m
           -> far cones disappear from masks + edges.
        2) Per-detection label: distance at bbox center for drawing text.
        """
        frame_vis = frame_bgr.copy()

        # --- Convert depth to numpy (raw units) ---
        depth_img = np.asanyarray(depth_frame.get_data())  # uint16
        # Convert to meters
        depth_m = depth_img * self.depth_scale

        # Build a depth-based mask: 255 where depth is valid and <= max_distance
        depth_mask = np.zeros_like(depth_img, dtype=np.uint8)
        depth_mask[(depth_m > 0) & (depth_m <= self.max_distance_m)] = 255

        # --- HSV + color masks ---
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        mask_y_raw = cv2.inRange(hsv, self.lowerYellow, self.upperYellow)
        mask_b_raw = cv2.inRange(hsv, self.lowerBlue,   self.upperBlue)

        # Apply depth gating to color masks (pixelwise AND)
        mask_y_raw = cv2.bitwise_and(mask_y_raw, depth_mask)
        mask_b_raw = cv2.bitwise_and(mask_b_raw, depth_mask)

        mask_y = self._clean_mask(mask_y_raw)
        mask_b = self._clean_mask(mask_b_raw)

        # === MERGED MASK (all cones, any of the two colors, and within depth range) === #
        mask_all = cv2.bitwise_or(mask_y, mask_b)

        # --- Canny on gated masks ---
        edges_y = cv2.Canny(mask_y, self.canny_t1, self.canny_t2)
        edges_b = cv2.Canny(mask_b, self.canny_t1, self.canny_t2)
        edges_all = cv2.Canny(mask_all, self.canny_t1, self.canny_t2)

        # Helper: get distance at bbox center (in meters)
        def center_distance_m(x, y, w, h):
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            d = depth_frame.get_distance(cx, cy)  # already in meters
            return float(d)

        # --- Contours from edges (yellow) ---
        contours_y, _ = cv2.findContours(edges_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_y:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / float(w) if w > 0 else 0.0
            # Cones are roughly “tall-ish”
            if aspect < 1.2:
                continue

            dist_m = center_distance_m(x, y, w, h)
            # Should already be within range due to depth_mask, but we keep this as a sanity check:
            if dist_m <= 0 or dist_m > self.max_distance_m:
                continue

            cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame_vis, f"Yellow cone {dist_m:.1f} m", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- Contours from edges (blue) ---
        contours_b, _ = cv2.findContours(edges_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_b:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / float(w) if w > 0 else 0.0
            if aspect < 1.2:
                continue

            dist_m = center_distance_m(x, y, w, h)
            if dist_m <= 0 or dist_m > self.max_distance_m:
                continue

            cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame_vis, f"Blue cone {dist_m:.1f} m", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        debug = {
            "depth_mask": depth_mask,   # where depth <= max_distance
            "mask_y": mask_y,
            "mask_b": mask_b,
            "mask_all": mask_all,
            "edges_y": edges_y,
            "edges_b": edges_b,
            "edges_all": edges_all
        }

        return frame_vis, debug


# ============== safe destroy helper so OpenCV doesn’t crash ============== #
def safe_destroy(win_name: str):
    try:
        cv2.destroyWindow(win_name)
    except cv2.error:
        # Window doesn't exist yet -> ignore
        pass


# ================================ Main ================================= #
def main():
    # RealSense RGB + depth; we ask for 1920x1080@30 but handler will pick closest valid
    cam = RealSenseRGBHandler(desired_color=(1920, 1080, 15))

    # Cone detector using HSV + Canny + contours + depth filter
    detector = ConeDetectorCanny(min_area=100,
                                 canny_thresh1=50,
                                 canny_thresh2=120,
                                 max_distance_m=4.2,           # ignore cones beyond 15 m
                                 depth_scale=cam.depth_scale)   # pass RealSense scale

    show_debug = False

    try:
        while True:
            ret, frame, depth_frame = cam.get_frame()
            if not ret:
                print("Failed to grab frame")
                break

            # Detect cones (and draw on the frame), depth-filtered at mask level
            vis, dbg = detector.process(frame, depth_frame)



            # Show main view
            cv2.imshow("RealSense RGB – Cones (Canny + HSV + depth gate)", vis)

            # Optional debug windows
            if show_debug:
                cv2.imshow("Depth mask (<= max dist)", dbg["depth_mask"])
                cv2.imshow("Mask – all cones",        dbg["mask_all"])
                cv2.imshow("Edges – all cones",       dbg["edges_all"])
            else:
                safe_destroy("Depth mask (<= max dist)")
                safe_destroy("Mask – all cones")
                safe_destroy("Edges – all cones")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_debug = not show_debug

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()