import signal
import sys
import time
import logging
from datetime import datetime
from perception.perception_module import PerceptionModule
from logic.logic_module import LogicModule
from control.arduino_interface import ArduinoInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'race_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for cleanup
perception = None
arduino = None
logic = None


class PerformanceMonitor:
    """Monitor loop performance and timing"""

    def __init__(self, window_size=30):
        self.times = []
        self.window_size = window_size
        self.last_time = time.time()

    def update(self):
        current_time = time.time()
        loop_time = current_time - self.last_time
        self.last_time = current_time

        self.times.append(loop_time)
        if len(self.times) > self.window_size:
            self.times.pop(0)

        return loop_time

    def get_stats(self):
        if not self.times:
            return None
        avg_time = sum(self.times) / len(self.times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        return {
            'avg_loop_time': avg_time,
            'avg_fps': avg_fps,
            'min_loop_time': min(self.times),
            'max_loop_time': max(self.times)
        }


class SafetyMonitor:
    """Monitor for safety violations and anomalies"""

    def __init__(self, max_steering=30.0, no_cone_timeout=5.0):
        self.max_steering = max_steering
        self.no_cone_timeout = no_cone_timeout
        self.last_cone_time = time.time()
        self.consecutive_no_cones = 0
        self.max_consecutive_no_cones = 15
        self.last_log_time = 0

    def check_steering(self, angle):
        """Check if steering angle is within safe limits"""
        if angle is not None and abs(angle) > self.max_steering:
            return min(max(angle, -self.max_steering), self.max_steering)
        return angle

    def check_cone_detection(self, cones_detected):
        """Monitor cone detection for potential issues"""
        if not cones_detected or len(cones_detected) == 0:
            self.consecutive_no_cones += 1

            if self.consecutive_no_cones >= self.max_consecutive_no_cones:
                elapsed = time.time() - self.last_cone_time
                if elapsed > self.no_cone_timeout:
                    if time.time() - self.last_log_time > 5.0:
                        logger.error(f"No cones detected for {elapsed:.1f}s - possible issue!")
                        self.last_log_time = time.time()
                    return False
        else:
            self.consecutive_no_cones = 0
            self.last_cone_time = time.time()
            self.last_log_time = 0

        return True


def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals"""
    logger.info("Shutdown signal received! Cleaning up...")

    # Stop motors immediately
    if arduino is not None:
        try:
            logger.info("Stopping motors...")
            arduino.send(0, 0)
            time.sleep(0.1)
            arduino.close()
            logger.info("Arduino connection closed.")
        except Exception as e:
            logger.error(f"Error closing Arduino: {e}")

    # Close camera
    if perception is not None:
        try:
            logger.info("Closing camera...")
            perception.shutdown()
            logger.info("Camera closed.")
        except Exception as e:
            logger.error(f"Error closing camera: {e}")

    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


def main():
    global perception, arduino, logic

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 60)
    logger.info("  Ctrl+C - Emergency stop")
    logger.info("=" * 60)

    # Initialize monitors
    perf_monitor = PerformanceMonitor()
    safety_monitor = SafetyMonitor(max_steering=30.0)

    # Initialize modules with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Initialization attempt {attempt + 1}/{max_retries}")

            logger.info("Starting perception module...")
            perception = PerceptionModule()

            logger.info("Starting logic module...")
            logic = LogicModule()

            logger.info("Starting Arduino interface...")
            arduino = ArduinoInterface("/dev/ttyACM0")

            logger.info("All modules initialized successfully!")
            break

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                logger.error("Max retries reached. Exiting.")
                if perception:
                    perception.shutdown()
                if arduino:
                    arduino.close()
                sys.exit(1)

    # Main control loop
    try:
        logger.info("Starting control loop...")
        loop_count = 0
        stats_interval = 30

        while True:
            loop_start = time.time()

            try:
                # Perception
                cones_world, img = perception.run()

                # Safety check
                if not safety_monitor.check_cone_detection(cones_world):
                    logger.warning("Extended cone detection failure - continuing with caution")

                # Logic
                blue, yellow = logic.cone_sorting(cones_world)
                midpoints = logic.cone_midpoints(blue, yellow, img)
                target = logic.Interpolation(midpoints)

                # Calculate control outputs
                if target:
                    angle = logic.steering_angle(target)
                    angle = safety_monitor.check_steering(angle)
                    speed = 32
                else:
                    if loop_count % 10 == 0:
                        logger.debug("No targets found")
                    angle = 0
                    speed = 32

                # Send to Arduino
                arduino.send(angle, speed)

                # Performance monitoring
                loop_time = perf_monitor.update()
                loop_count += 1

                # Periodic stats logging
                if loop_count % stats_interval == 0:
                    stats = perf_monitor.get_stats()
                    if stats:
                        logger.info(f"Performance: {stats['avg_fps']:.1f} FPS, "
                                    f"Loop time: {stats['avg_loop_time'] * 1000:.1f}ms "
                                    f"(min: {stats['min_loop_time'] * 1000:.1f}ms, "
                                    f"max: {stats['max_loop_time'] * 1000:.1f}ms)")
                        logger.info(f"Cones detected: {len(cones_world)}, "
                                    f"Blue: {len(blue)}, Yellow: {len(yellow)}")

            except Exception as e:
                logger.error(f"Error in control loop iteration: {e}")
                try:
                    arduino.send(0, 20)
                except:
                    pass
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Fatal error in control loop: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.info("Performing cleanup...")

        if arduino:
            try:
                logger.info("Stopping motors...")
                arduino.send(0, 0)
                time.sleep(0.2)
                arduino.close()
                logger.info("Arduino connection closed.")
            except Exception as e:
                logger.error(f"Error during Arduino cleanup: {e}")

        if perception:
            try:
                perception.shutdown()
                logger.info("Camera closed.")
            except Exception as e:
                logger.error(f"Error during camera cleanup: {e}")

        # Final stats
        stats = perf_monitor.get_stats()
        if stats:
            logger.info(f"Final statistics - Total loops: {loop_count}, "
                        f"Average FPS: {stats['avg_fps']:.1f}")

        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()