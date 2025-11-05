#!/usr/bin/env python3
"""
Simple test runner for the state-service tests.
Run with: python run_tests.py
"""

import subprocess
import sys
import os


def run_tests():
    """Run all tests using pytest"""
    try:
        # Change to the state-service directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Run pytest
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=False,
        )

        return result.returncode == 0

    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
