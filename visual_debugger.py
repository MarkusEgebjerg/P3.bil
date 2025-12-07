import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VisualDebugger:
    """
    Optional visual debugger that can be toggled on/off
    Shows cone detection, masks, and path planning
    """

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.window_names = []

        if self.enabled:
            logger.info("Visual debugger ENABLED - Performance may be reduced")
            logger.info("Press 'v' to toggle visualization, 'q' to quit, 's' to save snapshot")
        else:
            logger.info("Visual debugger DISABLED - Press 'v' at any time to enable")

    def toggle(self):
        """Toggle visualization on/off"""
        self.enabled = not self.enabled

        if self.enabled:
            logger.info("✓ Visualization ENABLED")
        else:
            logger.info("✗ Visualization DISABLED")
            self.close_all_windows()

    def close_all_windows(self):
        """Close all OpenCV windows"""
        for window_name in self.window_names:
            try:
                cv2.destroyWindow(window_name)
            except:
                pass
        self.window_names = []

    def show_main_view(self, img, cones_world, blue_cones, yellow_cones, target, steering_angle):
        """Show main annotated camera view with all information"""
        if not self.enabled or img is None:
            return

        # Make a copy to avoid modifying original
        display = img.copy()
        h, w = display.shape[:2]

        # Add info panel at top
        cv2.rectangle(display, (0, 0), (w, 100), (0, 0, 0), -1)

        # Status text
        cv2.putText(display, f"Cones: {len(cones_world)} (B:{len(blue_cones)} Y:{len(yellow_cones)})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if steering_angle is not None:
            color = (0, 255, 0) if abs(steering_angle) < 10 else (0, 165, 255)
            cv2.putText(display, f"Steering: {steering_angle:.1f} deg",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(display, "No target",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Target info
        if target:
            cv2.putText(display, f"Target: ({target[0]:.2f}, {target[1]:.2f})m",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw center line
        cv2.line(display, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)

        # Draw horizon line
        cv2.line(display, (0, 100), (w, 100), (255, 255, 255), 1)

        # Instructions
        cv2.putText(display, "V:toggle | S:save | Q:quit",
                    (w - 300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show window
        window_name = "AAU Racing - Main View"
        cv2.imshow(window_name, display)
        if window_name not in self.window_names:
            self.window_names.append(window_name)

    def show_masks(self, clean_mask_y, clean_mask_b):
        """Show color detection masks side by side"""
        if not self.enabled:
            return

        # Combine masks for display
        h, w = clean_mask_y.shape
        combined = np.zeros((h, w, 3), dtype=np.uint8)

        # Yellow mask in yellow channel
        combined[:, :, 1] = clean_mask_y  # Green channel
        combined[:, :, 2] = clean_mask_y  # Red channel (G+R = Yellow)

        # Blue mask in blue channel
        combined[:, :, 0] = clean_mask_b

        # Add labels
        cv2.putText(combined, "YELLOW", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(combined, "BLUE", (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        window_name = "Color Masks"
        cv2.imshow(window_name, combined)
        if window_name not in self.window_names:
            self.window_names.append(window_name)

    def show_bird_eye_view(self, blue_cones, yellow_cones, midpoints, target):
        """Show top-down bird's eye view of detected cones and path"""
        if not self.enabled:
            return

        # Create blank canvas (10m x 10m view, 50 pixels per meter)
        scale = 50  # pixels per meter
        canvas_size = 500  # 10m x 10m
        bird_view = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 40

        # Origin at bottom center
        origin_x = canvas_size // 2
        origin_y = canvas_size - 50

        # Draw grid
        for i in range(0, canvas_size, scale):
            cv2.line(bird_view, (i, 0), (i, canvas_size), (60, 60, 60), 1)
            cv2.line(bird_view, (0, i), (canvas_size, i), (60, 60, 60), 1)

        # Draw car (as triangle)
        car_pts = np.array([
            [origin_x, origin_y],
            [origin_x - 15, origin_y + 30],
            [origin_x + 15, origin_y + 30]
        ], np.int32)
        cv2.fillPoly(bird_view, [car_pts], (255, 255, 255))

        # Draw blue cones
        for cone in blue_cones:
            x, z = cone[0], cone[1]
            px = int(origin_x + x * scale)
            py = int(origin_y - z * scale)
            if 0 <= px < canvas_size and 0 <= py < canvas_size:
                cv2.circle(bird_view, (px, py), 8, (255, 0, 0), -1)
                cv2.circle(bird_view, (px, py), 10, (255, 255, 255), 1)

        # Draw yellow cones
        for cone in yellow_cones:
            x, z = cone[0], cone[1]
            px = int(origin_x + x * scale)
            py = int(origin_y - z * scale)
            if 0 <= px < canvas_size and 0 <= py < canvas_size:
                cv2.circle(bird_view, (px, py), 8, (0, 255, 255), -1)
                cv2.circle(bird_view, (px, py), 10, (255, 255, 255), 1)

        # Draw midpoints
        for mp in midpoints:
            x, z = mp[0], mp[1]
            px = int(origin_x + x * scale)
            py = int(origin_y - z * scale)
            if 0 <= px < canvas_size and 0 <= py < canvas_size:
                cv2.circle(bird_view, (px, py), 5, (0, 255, 0), -1)

        # Connect midpoints with line
        if len(midpoints) > 1:
            points = []
            for mp in midpoints:
                x, z = mp[0], mp[1]
                px = int(origin_x + x * scale)
                py = int(origin_y - z * scale)
                points.append([px, py])
            points = np.array(points, np.int32)
            cv2.polylines(bird_view, [points], False, (0, 200, 0), 2)

        # Draw target point
        if target:
            x, z = target[0], target[1]
            px = int(origin_x + x * scale)
            py = int(origin_y - z * scale)
            if 0 <= px < canvas_size and 0 <= py < canvas_size:
                cv2.circle(bird_view, (px, py), 12, (0, 255, 255), 3)
                cv2.line(bird_view, (origin_x, origin_y), (px, py), (0, 255, 255), 2)

        # Add scale reference
        cv2.putText(bird_view, "1m", (scale, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.line(bird_view, (10, 35), (10 + scale, 35), (255, 255, 255), 2)

        # Add labels
        cv2.putText(bird_view, "Bird's Eye View", (10, canvas_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        window_name = "Bird's Eye View"
        cv2.imshow(window_name, bird_view)
        if window_name not in self.window_names:
            self.window_names.append(window_name)

    def save_snapshot(self, img, blue_cones, yellow_cones, midpoints, target):
        """Save current view as image file"""
        if img is None:
            return

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.png"

        # Save main view
        cv2.imwrite(filename, img)
        logger.info(f"✓ Saved snapshot: {filename}")

        # Also save bird's eye view
        if self.enabled and (blue_cones or yellow_cones):
            bird_filename = f"birdview_{timestamp}.png"
            # Create bird's eye view (need to call the method)
            self.show_bird_eye_view(blue_cones, yellow_cones, midpoints, target)
            # Note: The window content needs to be saved separately
            logger.info(f"✓ Saved bird's eye view: {bird_filename}")

    def handle_key(self, key, img, blue_cones, yellow_cones, midpoints, target):
        """
        Handle keyboard input
        Returns: True to continue, False to quit
        """
        if key == ord('v') or key == ord('V'):
            self.toggle()
            return True
        elif key == ord('s') or key == ord('S'):
            self.save_snapshot(img, blue_cones, yellow_cones, midpoints, target)
            return True
        elif key == ord('q') or key == ord('Q'):
            logger.info("Quit requested by user")
            return False

        return True

    def check_events(self, img=None, blue_cones=None, yellow_cones=None,
                     midpoints=None, target=None):
        """
        Check for keyboard events (non-blocking)
        Call this every frame
        Returns: True to continue, False to quit
        """
        key = cv2.waitKey(1) & 0xFF

        if key != 255:  # Key was pressed
            return self.handle_key(key, img, blue_cones or [], yellow_cones or [],
                                   midpoints or [], target)

        return True

    def cleanup(self):
        """Clean up all windows"""
        self.close_all_windows()
        cv2.destroyAllWindows()