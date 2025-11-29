#!/usr/bin/env python3
"""
generate_index.py

Creates an HTML index of episode directories.  By default the script
produces `index.html`.  If the optional flag ``--track_by_track`` is
given, only episodes whose `keywords.json` file contains the keyword
``track_by_track_review`` are returned and the output file defaults to
``track_by_track.html``.

The HTML is modern and responsive – 3‑line‑wide grid, subtle hover
effects, and a pleasant gray/white color palette.  No external
dependencies – just standard library modules.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def list_episode_dirs(base_dir: Path) -> List[str]:
    """Return a sorted list of names of all non‑hidden sub‑directories inside `base_dir`."""
    try:
        entries = os.listdir(base_dir)
    except FileNotFoundError as exc:
        sys.stderr.write(f"Directory not found: {base_dir}\n")
        sys.exit(1)
    except PermissionError as exc:
        sys.stderr.write(f"Permission denied while reading: {base_dir}\n")
        sys.exit(1)

    dirs = sorted(
        name
        for name in entries
        if not name.startswith(".") and (base_dir / name).is_dir()
    )
    return dirs


def _load_key_words(json_path: Path) -> Optional[List[str]]:
    """Load the keyword list from `json_path`, if possible.
    Returns `None` on any error (missing file, JSON error, unexpected format).
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    return [str(x) for x in data]


def filter_by_keyword(
    episode_dirs: Iterable[str], base_path: Path, keyword: str
) -> List[str]:
    """Return the subset of `episode_dirs` whose `keywords.json` contains `keyword`."""
    filtered: List[str] = []
    for ep in episode_dirs:
        json_path = base_path / ep / "keywords.json"
        if not json_path.is_file():
            continue
        keywords = _load_key_words(json_path)
        if keywords and keyword in keywords:
            filtered.append(ep)
    return filtered


def build_html(
    entries: Iterable[str],
    base_path: Path,
    *,
    title: str = "Episode List",
    output_filename: str = "index.html",
) -> str:
    """Return the full HTML string."""
    header = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{
    font-family: system-ui, sans-serif;
    background: #f4f4f9;
    margin: 0;
    padding: 2rem;
  }}
  h1 {{ color: #222; text-align: center; margin-bottom: 1.5rem; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
  }}
  .card {{
    background: #fff;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.07);
    text-align: center;
  }}
  .card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    background: #e3f2fd;
  }}
  a {{ color: #0066cc; text-decoration: none; font-weight: 500; }}
  a:hover {{ text-decoration: none; color: #004494; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="grid">
"""

    # Convert entries to list if it's an iterator
    entries_list = list(entries) if not isinstance(entries, list) else entries
    body = ""
    for name in entries_list:
        # Try to compute relative path, fall back to absolute if not possible
        try:
            rel_base = base_path.relative_to(Path.cwd())
            href = os.path.join(rel_base, name, "episode.html").replace("\\", "/")
        except ValueError:
            # base_path is not relative to cwd, use absolute path or just the name
            href = os.path.join(base_path, name, "episode.html").replace("\\", "/")

        body += f'  <a class="card" href="{href}">{html.escape(name)}</a>\n'

    footer = """</div>
</body>
</html>"""

    return header + body + footer


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML index of episode directories."
    )
    parser.add_argument(
        "episodes_dir",
        type=Path,
        help="Path to the directory that contains episode sub‑directories",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output HTML filename (default: index.html or track_by_track.html)",
    )
    parser.add_argument(
        "--track_by_track",
        action="store_true",
        help='Include only episodes whose "keywords.json" contains "track_by_track_review"',
    )
    args = parser.parse_args()

    episodes_dir = args.episodes_dir
    if not episodes_dir.is_dir():
        sys.stderr.write(f"Error: {episodes_dir} is not a directory.\n")
        sys.exit(1)

    # 1. Discover episode directories
    episode_dirs = list_episode_dirs(episodes_dir)

    # 2. Optionally filter by keyword
    if args.track_by_track:
        episode_dirs = filter_by_keyword(
            episode_dirs, episodes_dir, "track_by_track_review"
        )
        default_output = "track_by_track.html"
        page_title = "Track‑By‑Track Review Episodes"
    else:
        default_output = "index.html"
        page_title = "Episode List"

    output_file = args.output or default_output

    # 3. Build the HTML
    html_content = build_html(
        episode_dirs,
        episodes_dir,
        title=page_title,
        output_filename=output_file,
    )

    # 4. Write the file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
    except OSError as exc:
        sys.stderr.write(f"Could not write to {output_file}: {exc}\n")
        sys.exit(1)

    # 5. Summary
    count = len(episode_dirs)
    if count == 0:
        print(f"Warning: No episodes found for the requested criteria.")
    print(f"Generated {output_file} with {count} episode link{'s' if count != 1 else ''}.")


if __name__ == "__main__":
    main()
