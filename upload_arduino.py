#!/usr/bin/env python3
import subprocess
import sys
import time
import os
import hashlib


def get_file_hash(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def find_arduino_port():
    possible_ports = [
        '/dev/ttyACM0',
        '/dev/ttyACM1',
        '/dev/ttyUSB0',
        '/dev/ttyUSB1'
    ]

    for port in possible_ports:
        if os.path.exists(port):
            print(f"Found Arduino at: {port}")
            return port

    print("✗ No Arduino found!")
    return None


def upload_arduino_sketch(sketch_path, port, board="arduino:avr:uno"):
    print("\n" + "=" * 30)
    print("ARDUINO AUTO-UPLOAD")
    print("=" * 30)

    # Check if sketch exists
    if not os.path.exists(sketch_path):
        print(f"✗ Error: Sketch not found at {sketch_path}")
        return False

    print(f"Sketch found: {sketch_path}")

    # Check if port exists
    if not port:
        port = find_arduino_port()
        if not port:
            print("✗ Cannot proceed without Arduino connection")
            return False

    hash_file = "/tmp/arduino_sketch.hash"
    current_hash = get_file_hash(sketch_path)

    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            last_hash = f.read().strip()

        if last_hash == current_hash:
            print("Sketch unchanged, skipping upload")
            print("  (Delete /tmp/arduino_sketch.hash to force upload)")
            return True

    print(f"Board: {board}")
    print(f"Port: {port}")
    print("\nCompiling sketch...")

    try:
        # Compile the sketch
        compile_cmd = [
            'arduino-cli', 'compile',
            '--fqbn', board,
            sketch_path
        ]

        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print("✗ Compilation failed!")
            print(result.stderr)
            return False

        print("Compilation successful!")

        # Upload the sketch
        print("\nUploading to Arduino...")
        upload_cmd = [
            'arduino-cli', 'upload',
            '--fqbn', board,
            '--port', port,
            sketch_path
        ]

        result = subprocess.run(
            upload_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print("✗ Upload failed!")
            print(result.stderr)
            return False

        print("Upload successful!")

        # Save hash to avoid re-uploading
        with open(hash_file, 'w') as f:
            f.write(current_hash)

        # Wait for Arduino to reset
        print("\nWaiting for Arduino to reset...")
        time.sleep(3)

        print("=" * 60)
        print("ARDUINO READY")
        print("=" * 60 + "\n")

        return True

    except subprocess.TimeoutExpired:
        print("✗ Upload timeout!")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Main function"""

    # Default values
    sketch_path = "/app/arduino/motorcontroller/motorcontroller.ino"
    port = None  # Auto-detect
    board = "arduino:avr:uno"

    # Parse command line arguments
    if len(sys.argv) > 1:
        sketch_path = sys.argv[1]
    if len(sys.argv) > 2:
        port = sys.argv[2]
    if len(sys.argv) > 3:
        board = sys.argv[3]

    # Upload sketch
    success = upload_arduino_sketch(sketch_path, port, board)

    if not success:
        print("\n⚠ Upload failed, but continuing anyway...")
        print("You may need to upload manually using:")
        print(f"  arduino-cli upload --fqbn {board} --port {port} {sketch_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()