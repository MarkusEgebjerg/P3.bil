#!/usr/bin/env python3

import sys
import time
import cv2
import argparse
from perception.perception_module import PerceptionModule
from logic.logic_module import LogicModule
from control.arduino_interface import ArduinoInterface


def test_camera():
    """Test camera feed and cone detection"""
    print("\n" + "=" * 60)
    print("CAMERA TEST")
    print("=" * 60)
    print("Press 'q' to quit, 's' to save snapshot\n")

    try:
        perception = PerceptionModule()
        logic = LogicModule()
        snapshot_count = 0

        while True:
            cones_world, img = perception.run()

            if img is not None:
                # Add info overlay
                cv2.putText(img, f"Cones detected: {len(cones_world)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show detected cones info
                blue = [c for c in cones_world if c[2] == "Blue"]
                yellow = [c for c in cones_world if c[2] == "Yellow"]
                cv2.putText(img, f"Blue: {len(blue)} Yellow: {len(yellow)}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Process with logic module
                midpoints = logic.cone_midpoints(blue, yellow, img)
                target = logic.Interpolation(midpoints)

                if target:
                    angle = logic.steering_angle(target)
                    cv2.putText(img, f"Steering: {angle:.1f} deg",
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Camera Test", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"snapshot_{snapshot_count:03d}.png"
                cv2.imwrite(filename, img)
                print(f"Saved {filename}")
                snapshot_count += 1

        perception.shutdown()
        print("\nCamera test complete!")

    except Exception as e:
        print(f"Error during camera test: {e}")
        return False

    return True


def test_arduino():
    """Test Arduino connection and motor control"""
    print("\n" + "=" * 60)
    print("ARDUINO TEST")
    print("=" * 60)
    print("Testing motor control - watch the RC car!\n")

    try:
        arduino = ArduinoInterface("/dev/ttyACM0")

        # Test sequence
        tests = [
            ("Center steering, no movement", 0, 0, 2),
            ("Center steering, slow forward", 0, 25, 2),
            ("Left steering, slow forward", -10, 25, 2),
            ("Right steering, slow forward", 10, 25, 2),
            ("Center steering, medium speed", 0, 32, 2),
            ("Stop", 0, 0, 1),
        ]

        for description, angle, speed, duration in tests:
            print(f"  {description} (angle={angle}°, speed={speed})")
            for _ in range(int(duration * 10)):
                arduino.send(angle, speed)
                time.sleep(0.1)

        # Final stats
        stats = arduino.get_stats()
        print(f"\nArduino Stats:")
        print(f"  Commands sent: {stats['commands_sent']}")
        print(f"  Errors: {stats['error_count']}")
        print(f"  Connected: {stats['is_connected']}")

        arduino.close()
        print("\nArduino test complete!")

    except Exception as e:
        print(f"Error during Arduino test: {e}")
        return False

    return True


def test_integration():
    """Test full system integration"""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST")
    print("=" * 60)
    print("Running full system for 10 seconds")
    print("Press Ctrl+C to stop early\n")

    try:
        perception = PerceptionModule()
        logic = LogicModule()
        arduino = ArduinoInterface("/dev/ttyACM0")

        start_time = time.time()
        loop_count = 0

        while time.time() - start_time < 10:
            cones_world, img = perception.run()

            blue, yellow = logic.cone_sorting(cones_world)
            midpoints = logic.cone_midpoints(blue, yellow, img)
            target = logic.Interpolation(midpoints)

            if target:
                angle = logic.steering_angle(target)
                speed = 32
                print(f"Loop {loop_count}: Cones={len(cones_world)}, "
                      f"Steering={angle:.1f}°, Speed={speed}")
            else:
                angle = 0
                speed = 32
                print(f"Loop {loop_count}: No target found")

            arduino.send(angle, speed)
            loop_count += 1

            if img is not None:
                cv2.imshow("Integration Test", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Stop and cleanup
        arduino.send(0, 0)
        time.sleep(0.2)
        arduino.close()
        perception.shutdown()

        elapsed = time.time() - start_time
        fps = loop_count / elapsed
        print(f"\nIntegration test complete!")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Loops: {loop_count}")
        print(f"  Average FPS: {fps:.1f}")

    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Error during integration test: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Test AAU Racing RC Car System')
    parser.add_argument('--camera', action='store_true', help='Test camera only')
    parser.add_argument('--arduino', action='store_true', help='Test Arduino only')
    parser.add_argument('--integration', action='store_true', help='Test full integration')
    parser.add_argument('--all', action='store_true', help='Run all tests')

    args = parser.parse_args()

    # If no arguments, run all tests
    if not any([args.camera, args.arduino, args.integration, args.all]):
        args.all = True

    print("\n" + "=" * 60)
    print("AAU RACING - SYSTEM TEST SUITE")
    print("=" * 60)

    results = {}

    if args.all or args.camera:
        results['camera'] = test_camera()

    if args.all or args.arduino:
        results['arduino'] = test_arduino()

    if args.all or args.integration:
        results['integration'] = test_integration()

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.upper()}: {status}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()