#!/usr/bin/env python
"""
Jetson PWM Pin Diagnostic Tool
Tests which pins support PWM on your Jetson board
"""

import Jetson.GPIO as GPIO
import time
import sys

class pinDiagnostic:
    def test_dual_pwm(pin1, pin2, freq=50, duration=5):
        """Test two pins for simultaneous PWM"""
        print(f"\n{'=' * 60}")
        print(f"Testing SIMULTANEOUS PWM on pins {pin1} and {pin2}")
        print(f"{'=' * 60}")

        try:
            # Setup both pins
            GPIO.setup(pin1, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(pin2, GPIO.OUT, initial=GPIO.LOW)
            print(f"✓ Both pins configured")

            # Create PWM objects
            pwm1 = GPIO.PWM(pin1, freq)
            pwm2 = GPIO.PWM(pin2, freq)
            print(f"✓ PWM objects created for both pins")

            # Start both PWM
            pwm1.start(0)
            pwm2.start(0)
            print(f"✓ Both PWM started")

            # Test alternating duty cycles
            print(f"\nTesting for {duration} seconds...")
            print(f"Pin {pin1} will ramp up while pin {pin2} ramps down")

            start_time = time.time()

            while (time.time() - start_time) < duration:
                elapsed = time.time() - start_time
                progress = (elapsed % 2) / 2  # 0 to 1 over 2 seconds

                duty1 = int(progress * 100)
                duty2 = int((1 - progress) * 100)

                pwm1.ChangeDutyCycle(duty1)
                pwm2.ChangeDutyCycle(duty2)

                print(f"  Pin {pin1}: {duty1:3d}% | Pin {pin2}: {duty2:3d}%", end='\r')
                time.sleep(0.05)

            print(f"\n✓ Dual PWM test PASSED")

            # Cleanup
            pwm1.stop()
            pwm2.stop()
            return True

        except Exception as e:
            print(f"\n✗ Dual PWM test FAILED: {e}")
            return False


    def test_single_pin(pin, freq=50, duration=5):

        """Test a single pin for PWM capability"""
        print(f"\n{'=' * 60}")
        print(f"Testing Pin {pin} at {freq} Hz")
        print(f"{'=' * 60}")

        try:
            # Setup pin
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            print(f"✓ Pin {pin} configured as output")

            # Create PWM object
            pwm = GPIO.PWM(pin, freq)
            print(f"✓ PWM object created for pin {pin}")

            # Start PWM
            pwm.start(0)
            print(f"✓ PWM started on pin {pin}")

            # Test duty cycle changes
            print(f"\nCycling duty cycle for {duration} seconds...")
            print("(Measure pin {pin} with multimeter/oscilloscope)")

            start_time = time.time()
            duty = 0
            direction = 1

            while (time.time() - start_time) < duration:
                pwm.ChangeDutyCycle(duty)
                print(f"  Duty: {duty:3d}%", end='\r')

                duty += direction * 10
                if duty >= 100:
                    duty = 100
                    direction = -1
                elif duty <= 0:
                    duty = 0
                    direction = 1

                time.sleep(0.3)

            print(f"\n✓ Pin {pin} PWM test PASSED")

            # Cleanup
            pwm.stop()
            return True

        except Exception as e:
            print(f"\n✗ Pin {pin} PWM test FAILED: {e}")
            return False


    def main():
        print("\n" + "=" * 60)
        print("JETSON PWM DIAGNOSTIC TOOL")
        print("=" * 60)
        print(f"Board Model: {GPIO.model}")
        print(f"GPIO Mode: BOARD")
        print("=" * 60)

        # Set mode
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)

        # Pins to test (common PWM-capable pins on Jetson Nano)
        test_pins = [32, 33]

        results = {}

        try:
            # Test each pin individually
            print("\n\n### PHASE 1: Individual Pin Tests ###")
            for pin in test_pins:
                results[pin] = test_single_pin(pin, freq=50, duration=3)
                time.sleep(1)
                GPIO.cleanup()
                GPIO.setmode(GPIO.BOARD)

            # Test simultaneous PWM
            print("\n\n### PHASE 2: Simultaneous PWM Test ###")
            dual_result = test_dual_pwm(32, 33, freq=50, duration=5)

            # Summary
            print("\n\n" + "=" * 60)
            print("TEST SUMMARY")
            print("=" * 60)

            for pin, result in results.items():
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"Pin {pin} (individual): {status}")

            status = "✓ PASS" if dual_result else "✗ FAIL"
            print(f"Pins 32+33 (simultaneous): {status}")

            # Recommendations
            print("\n" + "=" * 60)
            print("RECOMMENDATIONS")
            print("=" * 60)

            if results.get(32) and results.get(33) and dual_result:
                print("✓ Both pins support simultaneous PWM!")
                print("  Your motor driver should work.")
            elif results.get(32) and results.get(33):
                print("⚠ Pins work individually but NOT simultaneously!")
                print("  SOLUTION: Use one PWM pin + direction control")
                print("  OR use software PWM for the second pin")
            elif results.get(33):
                print("⚠ Only pin 33 supports hardware PWM")
                print("  SOLUTION: Use pin 33 for PWM, control direction with digital pins")
            else:
                print("✗ PWM not working properly")
                print("  Check your Jetson.GPIO installation")

        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")

        finally:
            GPIO.cleanup()
            print("\n" + "=" * 60)
            print("Cleanup complete")
            print("=" * 60 + "\n")

