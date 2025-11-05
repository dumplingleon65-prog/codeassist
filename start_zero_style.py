#!/usr/bin/env python3
"""
Start Zero Style App Script
Opens a new browser tab with the zero-style app using episode data.
Accepts command-line arguments for episode ID and timestep.
"""

import argparse
import webbrowser as browser
import sys
import urllib.parse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a zero-style simulation using episode data"
    )
    parser.add_argument(
        "--episode",
        type=str,
        required=True,
        help="Episode ID (directory name in state-service/episodes)",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        required=True,
        help="Timestep number (line number in JSONL file, 1-indexed)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3003,
        help="Port number where the zero-style UI is running (default: 3003)",
    )
    parser.add_argument(
        "--episodes-dir",
        type=str,
        default="./persistent-data/state-service/episodes",
        help="Path to episodes directory (default: ./persistent-data/state-service/episodes)",
    )
    parser.add_argument(
        "--max-assistant-actions",
        type=int,
        default=2,
        help="Number of assistant actions to allow before stopping the simulation",
    )
    parser.add_argument(
        "--max-human-actions",
        type=int,
        default=1,
        help="Number of human policy actions to run after assistant actions",
    )
    parser.add_argument(
        "--assistant-noise-prob",
        type=float,
        default=0.05,
        help="Probability of replacing an assistant action with a random exploratory action",
    )
    parser.add_argument(
        "--assistant-noise-top-k",
        type=int,
        default=3,
        help="When exploration triggers, sample uniformly from the assistant policy's top-k actions",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        if args.max_assistant_actions <= 0:
            raise ValueError("--max-assistant-actions must be a positive integer")
        if args.max_human_actions < 0:
            raise ValueError("--max-human-actions must be zero or a positive integer")
        if args.assistant_noise_top_k <= 0:
            raise ValueError("--assistant-noise-top-k must be a positive integer")
        # Just pass episode and timestep - let the app load the data
        params = {
            "episode": args.episode,
            "timestep": str(args.timestep),
            "maxAssistantActions": str(args.max_assistant_actions),
            "maxHumanActions": str(args.max_human_actions),
            "assistantNoiseProb": str(args.assistant_noise_prob),
            "assistantNoiseTopK": str(args.assistant_noise_top_k),
        }

        # Build URL
        query_string = urllib.parse.urlencode(params)
        url = f"http://localhost:{args.port}?{query_string}"

        print("🎯 Starting zero-style simulation from episode data...")
        print(f"📍 Opening browser to http://localhost:{args.port}")
        print(f"📁 Episode: {args.episode}")
        print(f"⏱️  Timestep: {args.timestep}")
        print("")

        try:
            browser.open(url)
            print("✅ Browser opened!")
            print(
                f"🎯 Zero-style simulation will run with episode data from timestep {args.timestep}"
            )
            print("")
        except Exception as e:
            print(f"❌ Error opening browser: {e}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
