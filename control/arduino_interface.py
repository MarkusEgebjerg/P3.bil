import serial
import time
import logging

try:
    from config import CONTROL_CONFIG, SAFETY_CONFIG
except ImportError:
    CONTROL_CONFIG = {
        'default_speed': 32,
        'max_steering_angle': 30.0,
        'arduino_port': '/dev/ttyACM0',
        'arduino_baud': 115200,
        'command_delay': 0.01
    }
    SAFETY_CONFIG = {'max_steering_angle': 30.0}

logger = logging.getLogger(__name__)


class ArduinoInterface:
    def __init__(self, port=None, baud=None, timeout=0.05):
        self.port = port if port is not None else CONTROL_CONFIG['arduino_port']
        self.baud = baud if baud is not None else CONTROL_CONFIG['arduino_baud']
        self.timeout = timeout

        self.max_steering = SAFETY_CONFIG['max_steering_angle']
        self.command_delay = CONTROL_CONFIG['command_delay']

        self.ser = None
        self.last_command = (0, 0)
        self.command_count = 0
        self.error_count = 0
        self.ack_failures = 0  # Track failed acknowledgments
        self.last_ack_check = time.time()

        logger.info(f"Initializing Arduino on {self.port} at {self.baud} baud")
        self._connect()

    def _connect(self):
        """Connect to Arduino with retry logic"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logger.info(f"Connecting to Arduino (attempt {attempt + 1}/{max_attempts})...")
                self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
                time.sleep(2.5)  # Wait for Arduino reset

                # Clear any startup messages
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()

                # Wait for READY signal
                ready_timeout = time.time() + 3.0
                while time.time() < ready_timeout:
                    if self.ser.in_waiting > 0:
                        line = self.ser.readline().decode().strip()
                        logger.info(f"Arduino: {line}")
                        if "READY" in line:
                            break
                    time.sleep(0.1)

                logger.info("Arduino connected and ready!")

                # Send initial safe command
                self.send(0, 0)
                return

            except serial.SerialException as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1)
                else:
                    raise Exception(f"Failed to connect after {max_attempts} attempts")

    def send(self, steering, speed):
        if self.ser is None or not self.ser.is_open:
            logger.error("Arduino not connected!")
            return False

        try:
            # Validate and clamp inputs
            steering = max(-self.max_steering, min(self.max_steering, steering))
            speed = max(0, min(255, int(speed)))

            # Convert to protocol format
            steering_int = int(steering * 10)

            # Format message
            msg = f"{steering_int},{speed}\n"

            # Send command
            self.ser.write(msg.encode())
            self.command_count += 1
            self.last_command = (steering, speed)

            # Check for acknowledgment every 30 commands
            # This prevents overwhelming the serial buffer
            if self.command_count % 30 == 0:
                ack_received = False
                check_start = time.time()

                while time.time() - check_start < 0.01:  # 10ms window
                    if self.ser.in_waiting > 0:
                        try:
                            line = self.ser.readline().decode().strip()
                            if line:
                                logger.debug(f"Arduino: {line}")
                                ack_received = True
                                break
                        except:
                            pass

                if not ack_received:
                    self.ack_failures += 1

                    # Log health check
                    if time.time() - self.last_ack_check > 5.0:
                        logger.info(f"Arduino health: {self.command_count} cmds, "
                                    f"{self.ack_failures} ack fails, "
                                    f"{self.error_count} errors")
                        self.last_ack_check = time.time()

            return True

        except serial.SerialException as e:
            self.error_count += 1
            logger.error(f"Serial error: {e}")

            # Reconnect after multiple errors
            if self.error_count >= 5:
                logger.warning("Multiple errors - attempting reconnection...")
                self._reconnect()

            return False

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False

    def _reconnect(self):
        try:
            logger.info("Closing existing connection...")
            if self.ser:
                self.ser.close()
            time.sleep(1)

            logger.info("Attempting reconnection...")
            self._connect()
            self.error_count = 0
            logger.info("Reconnection successful!")

        except Exception as e:
            logger.error(f" Reconnection failed: {e}")

    def emergency_stop(self):
        logger.warning("EMERGENCY STOP!")
        for _ in range(5):  # Send multiple times
            self.send(0, 0)
            time.sleep(0.02)

    def test_communication(self):
        logger.info("Testing Arduino communication...")

        test_commands = [
            (0, 0),
            (10, 32),
            (-10, 32),
            (0, 0)
        ]

        success_count = 0
        for angle, speed in test_commands:
            if self.send(angle, speed):
                success_count += 1
            time.sleep(0.1)

        logger.info(f"Communication test: {success_count}/{len(test_commands)} successful")
        return success_count == len(test_commands)

    def get_stats(self):
        return {
            'commands_sent': self.command_count,
            'error_count': self.error_count,
            'ack_failures': self.ack_failures,
            'success_rate': (self.command_count - self.error_count) / max(self.command_count, 1) * 100,
            'last_command': self.last_command,
            'is_connected': self.ser is not None and self.ser.is_open
        }

    def close(self):
        if self.ser and self.ser.is_open:
            try:
                logger.info("Sending final stop command...")
                self.emergency_stop()
                time.sleep(0.2)

                # Log final stats
                stats = self.get_stats()
                logger.info(f"Final Arduino stats: {stats['commands_sent']} commands, "
                            f"{stats['success_rate']:.1f}% success rate")

                logger.info("Closing serial connection")
                self.ser.close()
                logger.info("Arduino closed successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        else:
            logger.warning("Arduino was not connected")