import signal
import sys
import time
import logging
from datetime import datetime
from collections import deque

# FIXED: Corrected import paths
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

# Global variables for cleanup - FIXED: Added shutdown flags
perception = None
arduino = None
logic = None
shutdown_complete = False

try:
    from config import CONTROL_CONFIG
except ImportError:
    CONTROL_CONFIG = {
        'default_speed': 40,  # PWM value (0-255)
        'max_steering_angle': 30,  # degrees
        'arduino_port': '/dev/ttyACM0',
        'arduino_baud': 115200,
        'command_delay': 0.01,
    }


# ============================================================================
# MODULE Hz PROFILING CLASSES
# ============================================================================

class ModuleTimer:
    """Timer for individual module Hz performance tracking"""

    def __init__(self, module_name, window_size=30):
        self.module_name = module_name
        self.window_size = window_size
        self.execution_times = deque(maxlen=window_size)
        self.start_time = None
        self.total_calls = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            return
        execution_time = time.time() - self.start_time
        self.execution_times.append(execution_time)
        self.total_calls += 1
        self.start_time = None
        return execution_time

    def get_stats(self):
        if not self.execution_times:
            return None
        times = list(self.execution_times)
        avg_time = sum(times) / len(times)
        avg_hz = 1.0 / avg_time if avg_time > 0 else 0
        return {
            'module': self.module_name,
            'avg_time_ms': avg_time * 1000,
            'avg_hz': avg_hz,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'total_calls': self.total_calls
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class SystemProfiler:
    """Profiles all modules - measures Hz performance"""

    def __init__(self, window_size=30, log_interval=30):
        self.timers = {}
        self.window_size = window_size
        self.log_interval = log_interval
        self.system_start_time = time.time()
        self.loop_count = 0

    def get_timer(self, module_name):
        if module_name not in self.timers:
            self.timers[module_name] = ModuleTimer(module_name, self.window_size)
        return self.timers[module_name]

    def time_module(self, module_name):
        return self.get_timer(module_name)

    def log_stats(self, force=False):
        self.loop_count += 1
        if not force and self.loop_count % self.log_interval != 0:
            return

        logger.info("=" * 80)
        logger.info(f"MODULE Hz PERFORMANCE REPORT (Loop #{self.loop_count})")
        logger.info("=" * 80)

        total_time = 0
        all_stats = []

        for module_name, timer in self.timers.items():
            stats = timer.get_stats()
            if stats:
                all_stats.append(stats)
                total_time += stats['avg_time_ms']

        all_stats.sort(key=lambda x: x['avg_time_ms'], reverse=True)

        for stats in all_stats:
            logger.info(
                f"{stats['module']:20s} | "
                f"Avg: {stats['avg_time_ms']:6.2f}ms ({stats['avg_hz']:5.1f} Hz) | "
                f"Min: {stats['min_time_ms']:6.2f}ms | "
                f"Max: {stats['max_time_ms']:6.2f}ms | "
                f"Calls: {stats['total_calls']:6d}"
            )

        logger.info("-" * 80)
        if total_time > 0:
            max_theoretical_hz = 1000.0 / total_time
            logger.info(f"Total Module Time: {total_time:.2f}ms per loop")
            logger.info(f"Max Theoretical System Hz: {max_theoretical_hz:.1f} Hz")

        uptime = time.time() - self.system_start_time
        logger.info(f"System Uptime: {uptime:.1f}s | Total Loops: {self.loop_count}")
        logger.info("=" * 80)


# ============================================================================


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
    global perception, arduino, logic, shutdown_complete

    if shutdown_complete:
        logger.warning("Shutdown already in progress, forcing exit...")
        sys.exit(1)

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
    shutdown_complete = True
    sys.exit(0)


def main():
    global perception, arduino, logic, shutdown_complete

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 30)
    logger.info("  Ctrl+C - Emergency stop")
    logger.info("=" * 30)

    # Initialize monitors
    perf_monitor = PerformanceMonitor()
    safety_monitor = SafetyMonitor(max_steering=30.0)

    # Initialize Hz profiler
    profiler = SystemProfiler(window_size=30, log_interval=30)
    logger.info("Module Hz profiler initialized")

    # Initialize modules with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Initialization attempt {attempt + 1}/{max_retries}")

            logger.info("Starting perception module...")
            perception = PerceptionModule(record_video=True, output_path="/mnt/user-data/outputs")

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
                    try:
                        perception.shutdown()
                    except:
                        pass
                if arduino:
                    try:
                        arduino.close()
                    except:
                        pass
                sys.exit(1)

    # Main control loop
    try:
        logger.info("Starting control loop with Hz profiling...")
        loop_count = 0
        stats_interval = 30

        while True:
            try:
                # PERCEPTION MODULE - Timed
                with profiler.time_module("Perception"):
                    cones_world, img = perception.run()

                # Safety check
                if not safety_monitor.check_cone_detection(cones_world):
                    logger.warning("Extended cone detection failure - continuing with caution")

                # LOGIC MODULE - Timed with sub-components
                with profiler.time_module("Logic_Sorting"):
                    blue, yellow = logic.cone_sorting(cones_world)

                with profiler.time_module("Logic_Midpoints"):
                    midpoints = logic.cone_midpoints(blue, yellow, img)

                with profiler.time_module("Logic_Interpolation"):
                    target = logic.Interpolation(midpoints)

                # Calculate control outputs
                angle = None
                speed = 0

                with profiler.time_module("Logic_Steering"):
                    if target:
                        angle = logic.steering_angle(target)
                        angle = safety_monitor.check_steering(angle)
                        speed = CONTROL_CONFIG['default_speed']
                    else:
                        if loop_count % 10 == 0:
                            logger.debug("No targets found")
                        angle = 0
                        speed = 0

                # MOTION CONTROL MODULE - Timed
                with profiler.time_module("Arduino"):
                    arduino.send(angle, speed)

                # Performance monitoring
                loop_time = perf_monitor.update()
                loop_count += 1

                # Log Hz profiler stats (every 30 loops by default)
                profiler.log_stats()

                # Periodic overall stats logging
                if loop_count % stats_interval == 0:
                    stats = perf_monitor.get_stats()
                    if stats:
                        logger.info("=" * 80)
                        logger.info(f"OVERALL SYSTEM (Loop #{loop_count})")
                        logger.info(f"System Hz: {stats['avg_fps']:.1f} Hz | "
                                    f"Loop Time: {stats['avg_loop_time'] * 1000:.1f}ms | "
                                    f"Cones: {len(cones_world)} (Blue: {len(blue)}, Yellow: {len(yellow)})")
                        logger.info("=" * 80)

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
        # Cleanup - FIXED: Only run once
        if not shutdown_complete:
            logger.info("Performing cleanup...")

            # Log final Hz profiler stats
            logger.info("\n" + "=" * 80)
            logger.info("FINAL MODULE Hz PERFORMANCE SUMMARY")
            logger.info("=" * 80)
            profiler.log_stats(force=True)

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
                            f"Average System Hz: {stats['avg_fps']:.1f}")

            logger.info("Shutdown complete.")
            shutdown_complete = True


if __name__ == "__main__":
    main()