#!/usr/bin/env python3
"""
transcribe_episodes.py
----------------------
Transcribe all audio files in episode directories using whisper-cli.exe.

For each episode folder in OUTPUT_ROOT, this script:
1. Finds the audio/video file
2. Runs whisper-cli.exe to transcribe it
3. Saves the transcription as a .txt file with the same base name

Command format:
{WHISPER_PATH}/whisper-cli.exe -m {MODELS_PATH}/ggml-medium-q8_0.bin -t 6 -otxt {INPUT_FILENAME} > {OUTPUT_FILENAME}
"""

import json
import logging
import os
import pathlib
import subprocess
import sys

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler('transcribe_output.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.propagate = False

# Configuration - UPDATE THESE PATHS FOR YOUR SYSTEM
OUTPUT_ROOT = pathlib.Path("t:/")

# Windows example: pathlib.Path("C:/whisper")
# macOS/Linux example: pathlib.Path("/usr/local/bin")
WHISPER_PATH = pathlib.Path("C:/dev/whisper.cpp")  # TODO: Update this path

# Windows example: pathlib.Path("C:/whisper/models")
# macOS/Linux example: pathlib.Path("/usr/local/share/whisper/models")
MODELS_PATH = pathlib.Path("C:/dev/whisper.cpp/models")  # TODO: Update this path

WHISPER_EXECUTABLE = WHISPER_PATH / "whisper-cli.exe"
MODEL_FILE = MODELS_PATH / "ggml-medium-q8_0.bin"
THREADS = 6

# Audio/video file extensions to process
MEDIA_EXTENSIONS = {".mp3", ".m4a", ".mp4", ".wav", ".ogg", ".flac", ".aac", ".webm", ".mov", ".avi"}


def find_media_file(episode_dir: pathlib.Path) -> pathlib.Path | None:
    """
    Find the first audio/video file in the episode directory.
    
    Args:
        episode_dir: Path to the episode directory
        
    Returns:
        Path to the media file, or None if not found
    """
    for file in episode_dir.iterdir():
        if file.is_file() and file.suffix.lower() in MEDIA_EXTENSIONS:
            return file
    return None


def transcription_exists(media_file: pathlib.Path) -> bool:
    """
    Check if a transcription file already exists for the given media file.
    
    Args:
        media_file: Path to the media file
        
    Returns:
        True if transcription exists, False otherwise
    """
    transcript_file = media_file.with_suffix('.txt')
    return transcript_file.exists() and transcript_file.stat().st_size > 0


def transcribe_file(media_file: pathlib.Path) -> bool:
    """
    Transcribe a media file using whisper-cli.exe.
    
    Args:
        media_file: Path to the media file to transcribe
        
    Returns:
        True if transcription succeeded, False otherwise
    """
    output_file = media_file.with_suffix('.txt')
    
    # Check if transcription already exists
    if transcription_exists(media_file):
        logger.info(f"SKIP - Transcription already exists: {output_file.name}")
        return True
    
    logger.info(f"Transcribing: {media_file.name}")
    
    # Build the command
    cmd = [
        str(WHISPER_EXECUTABLE),
        "-m", str(MODEL_FILE),
        "-t", str(THREADS),
        "-otxt",
        str(media_file)
    ]
    
    try:
        # Run whisper-cli and capture output
        with open(output_file, 'w', encoding='utf-8') as outf:
            result = subprocess.run(
                cmd,
                stdout=outf,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
        
        if result.returncode == 0:
            logger.info(f"SUCCESS - Transcription saved to: {output_file.name}")
            return True
        else:
            logger.error(f"FAILED - Whisper returned error code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            # Remove the output file if it was created but the process failed
            if output_file.exists():
                output_file.unlink()
            return False
            
    except FileNotFoundError:
        logger.error(f"ERROR - Whisper executable not found: {WHISPER_EXECUTABLE}")
        logger.error(f"Please update WHISPER_PATH in the script configuration")
        return False
    except Exception as exc:
        logger.exception(f"ERROR - Failed to transcribe {media_file.name}: {exc}")
        # Remove the output file if it was created but an error occurred
        if output_file.exists():
            output_file.unlink()
        return False


def main() -> None:
    """
    Main function to iterate over all episode directories and transcribe media files.
    """
    logger.info("=" * 70)
    logger.info("Starting transcription process")
    logger.info("=" * 70)
    
    # Verify paths exist
    if not OUTPUT_ROOT.exists():
        logger.error(f"Episode directory does not exist: {OUTPUT_ROOT}")
        sys.exit(1)
    
    if not WHISPER_EXECUTABLE.exists():
        logger.error(f"Whisper executable not found: {WHISPER_EXECUTABLE}")
        logger.error("Please update WHISPER_PATH in the script configuration")
        sys.exit(1)
    
    if not MODEL_FILE.exists():
        logger.error(f"Model file not found: {MODEL_FILE}")
        logger.error("Please update MODELS_PATH in the script configuration")
        sys.exit(1)
    
    logger.info(f"Whisper executable: {WHISPER_EXECUTABLE}")
    logger.info(f"Model file: {MODEL_FILE}")
    logger.info(f"Threads: {THREADS}")
    logger.info(f"Episode directory: {OUTPUT_ROOT}")
    logger.info("=" * 70)
    
    # Get all episode directories
    episode_dirs = [d for d in OUTPUT_ROOT.iterdir() if d.is_dir()]
    logger.info(f"Found {len(episode_dirs)} episode directories")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    no_media_count = 0
    
    for i, episode_dir in enumerate(episode_dirs, start=1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing episode {i}/{len(episode_dirs)}: {episode_dir.name}")
        logger.info(f"{'='*70}")
        
        # Find the media file
        media_file = find_media_file(episode_dir)
        
        if media_file is None:
            logger.warning(f"No media file found in: {episode_dir.name}")
            no_media_count += 1
            continue
        
        logger.info(f"Media file: {media_file.name}")
        
        # Check if already transcribed
        if transcription_exists(media_file):
            skip_count += 1
            logger.info(f"SKIP - Transcription already exists")
            continue
        
        # Transcribe the file
        if transcribe_file(media_file):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Transcription process complete")
    logger.info("=" * 70)
    logger.info(f"Total episodes: {len(episode_dirs)}")
    logger.info(f"Successfully transcribed: {success_count}")
    logger.info(f"Already transcribed (skipped): {skip_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"No media file: {no_media_count}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

