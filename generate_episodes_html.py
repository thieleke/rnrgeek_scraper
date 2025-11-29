#!/usr/bin/env python3
"""
generate_html.py
----------------
Standalone script to generate pretty HTML files for all episodes.

This script calls the generate_all_html() function from transcribe_episodes.py
to process all episodes and create episode.html files based on their metadata,
descriptions, and AI summaries.

Usage:
    python generate_html.py

The script will:
- Read metadata.json for episode name and date
- Read description.txt for episode description
- Read ai_show_summary.md for AI-generated summary (if available)
- Generate a styled HTML file with all content
"""

import pathlib
import sys

# Add the current directory to the path to ensure transcribe_episodes can be imported
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

try:
    from transcribe_episodes import generate_all_html, OUTPUT_ROOT
except ImportError:
    print("Error: Could not import from transcribe_episodes.py.")
    print("Please ensure this script is in the same directory as transcribe_episodes.py.")
    sys.exit(1)


# --- Configuration ---
# By default, this script uses the OUTPUT_ROOT defined in transcribe_episodes.py.
# If you want to process a different directory, uncomment and set the path below.
# EPISODES_DIR = pathlib.Path("path/to/your/episodes")
EPISODES_DIR = pathlib.Path("episodes")


if __name__ == "__main__":
    print("Starting HTML generation for all episodes...")
    print(f"Target episodes directory: {EPISODES_DIR.resolve()}")
    print("-" * 70)

    if not EPISODES_DIR.exists():
        print(f"Error: Episodes directory not found at '{EPISODES_DIR.resolve()}'")
        print("Please check the EPISODES_DIR path in this script or OUTPUT_ROOT in transcribe_episodes.py.")
        sys.exit(1)

    try:
        generate_all_html(root_dir=EPISODES_DIR)
        print("\nHTML generation complete!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

