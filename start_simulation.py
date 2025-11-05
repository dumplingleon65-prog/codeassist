#!/usr/bin/env python3
"""
Start Simulation Script
Opens a new browser tab with the simulation app and clears localStorage.
Accepts command-line arguments for interval time and total duration.
"""

import argparse
import webbrowser as browser
import sys


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a new simulation with configurable duration and refresh interval"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Total simulation duration in seconds (default: 3600)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=180,
        help="Refresh interval in seconds (default: 180)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3002,
        help="Port number where the web UI is running (default: 3002)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Convert minutes to milliseconds for the URL parameters
    duration_ms = args.duration * 1000
    interval_ms = args.interval * 1000

    # Build URL with parameters
    url = (
        f"http://localhost:{args.port}?"
        f"reset=true&"
        f"duration={duration_ms}&"
        f"interval={interval_ms}"
    )

    print("🚀 Starting new simulation...")
    print(f"📍 Opening browser to http://localhost:{args.port}")
    print(f"⏱️  Total duration: {args.duration} minutes")
    print(f"🔄 Refresh interval: {args.interval} minutes")
    print("")

    try:
        browser.open(url)
        print("✅ Browser opened!")
        print(
            f"🎯 Simulation will run for {args.duration} seconds with auto-refresh every {args.interval} seconds"
        )
        print("")
    except Exception as e:
        print(f"❌ Error opening browser: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
