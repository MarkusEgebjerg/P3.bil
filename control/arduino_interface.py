import serial
import time
import logging

# Import config
try:
    from config import CONTROL_CONFIG, SAFETY_CONFIG
except ImportError:
    # Fallback to default values if config.py not found
    CONTROL_CONFIG = {
        'default_speed': 32,
        'max_steering_angle': 30.0,
        'arduino_port': '/dev/ttyACM0',
        'arduino_baud': 115200,
        'command_delay': 0.01
    }
    SAFETY_CONFIG = {
        'max_steering_angle': 30.0
    }

logger = logging.getLogger(__name__)


class ArduinoInterface:
    """Enhanced Arduino interface with error handling and reconnection"""

    def __init__(self, port=None, baud=None, timeout=0.1):
        """
        Initialize Arduino interface

        Args:
            port: Serial port (uses config if None)
            baud: Baud rate (uses config if None)
            timeout: Serial timeout in seconds
        """
        # Use provided values or fall back to config
        self.port = port if port is not None else CONTROL_CONFIG['arduino_port']
        self.baud = baud if baud is not None else CONTROL_CONFIG['arduino_baud']
        self.timeout = timeout

        # Use config values
        self.max_steering = SAFETY_CONFIG['max_steering_angle']
        self.command_delay = CONTROL_CONFIG['command_delay']

        self.ser = None
        self.last_command = (0, 0)
        self.command_count = 0
        self.error_count = 0

        logger.info(f"Initializing Arduino on {self.port} at {self.baud} baud")
        self._connect()

    def _connect(self):
        """Connect to Arduino with retry logic"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logger.info(f"Attempting to connect to Arduino (attempt {attempt + 1}/{max_attempts})...")
                self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
                time.sleep(2)  # Wait for Arduino to reset
                logger.info("Arduino connected successfully!")

                # Send initial safe command
                self.send(0, 0)
                return

            except serial.SerialException as e:
                logger.warning(f"Connection attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1)
                else:
                    raise Exception(f"Failed to connect to Arduino after {max_attempts} attempts")

    def send(self, steering, speed):
        """
        Send steering and speed commands to Arduino

        Args:
            steering: Steering angle in degrees (clamped to config max)
            speed: Speed value (0-255)

        Returns:
            Response from Arduino or None
        """
        if self.ser is None or not self.ser.is_open:
            logger.error("Arduino not connected!")
            return None

        try:
            # Validate and clamp inputs using config values
            steering = max(-self.max_steering, min(self.max_steering, steering))
            speed = max(0, min(255, int(speed)))

            # Convert steering to integer (multiply by 10 for precision)
            steering_int = int(steering * 10)  # e.g., -300 to 300

            # Format message
            msg = f"{steering_int},{speed}\n"

            # Send command
            self.ser.write(msg.encode())
            self.command_count += 1
            self.last_command = (steering, speed)

            # Small delay to prevent Arduino overload (from config)
            #time.sleep(self.command_delay)

            # Try to read response
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode().strip()
                    if line:
                        # Only log every 30th response to reduce spam
                        if self.command_count % 30 == 0:
                            logger.debug(f"Arduino response: {line}")
                        return line
            except Exception as e:
                logger.debug(f"Error reading Arduino response: {e}")

            return None

        except serial.SerialException as e:
            self.error_count += 1
            logger.error(f"Serial communication error: {e}")

            # Attempt reconnection after multiple errors
            if self.error_count >= 5:
                logger.warning("Multiple errors detected, attempting reconnection...")
                self._reconnect()

            return None

        except Exception as e:
            logger.error(f"Unexpected error in send(): {e}")
            return None

    def _reconnect(self):
        """Attempt to reconnect to Arduino"""
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
            logger.error(f"Reconnection failed: {e}")

    def emergency_stop(self):
        """Send emergency stop command"""
        logger.warning("EMERGENCY STOP!")
        for _ in range(3):  # Send multiple times to ensure receipt
            self.send(0, 0)
            time.sleep(0.05)

    def get_stats(self):
        """Get statistics about Arduino communication"""
        return {
            'commands_sent': self.command_count,
            'error_count': self.error_count,
            'last_command': self.last_command,
            'is_connected': self.ser is not None and self.ser.is_open
        }

    def close(self):
        """Close the Arduino connection"""
        if self.ser and self.ser.is_open:
            try:
                # Send stop command before closing
                logger.info("Sending final stop command...")
                self.emergency_stop()
                time.sleep(0.1)

                logger.info("Closing serial connection...")
                self.ser.close()
                logger.info("Arduino connection closed successfully.")
            except Exception as e:
                logger.error(f"Error closing Arduino: {e}")
        else:
            logger.warning("Arduino was not connected or already closed.")