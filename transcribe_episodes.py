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
import re
import subprocess
import sys
import requests

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

FFMPEG_EXECUTABLE = WHISPER_PATH / "ffmpeg.exe"

# OpenAI-compatible API Configuration for AI Summarization
# TODO: Update these settings for your LLM server
OPENAI_API_URL = "http://10.0.0.1:1234/v1/chat/completions"  # LM Studio default
OPENAI_API_KEY = ""  # Some servers require this, even if it's a placeholder
OPENAI_MODEL = "local-model"  # Model name (often ignored by local servers)
OPENAI_MAX_TOKENS = 65535  # Maximum tokens for the summary response
OPENAI_TEMPERATURE = 0.5  # Temperature for response generation (0.0 = deterministic, 1.0 = creative)

# Audio/video file extensions to process
MEDIA_EXTENSIONS = {".mp3", ".m4a", ".mp4", ".wav", ".ogg", ".flac", ".aac", ".webm", ".mov", ".avi", ".m4v"}
VIDEO_EXTENSIONS = {".mp4", ".m4a", ".mov", ".avi", ".m4v"}


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
    transcript_file = media_file.parent / (media_file.stem + '.txt')
    return transcript_file.exists() and transcript_file.stat().st_size > 0


def detect_ai_transcription_stutter_error(transcript_file: pathlib.Path, max_duplicate_lines: int = 50) -> bool:
    """
    Detect AI transcription errors by checking for excessive duplicate lines.

    The function parses lines with timestamp headers in the format:
    [HH:MM:SS.mmm --> HH:MM:SS.mmm]   text content

    It skips the first 32 characters (timestamp header) and counts duplicate
    content lines. If more than max_duplicate_lines unique content strings
    appear more than once, it flags the transcription as having errors.

    Args:
        transcript_file: Path to the transcription file
        max_duplicate_lines: Maximum allowed duplicate lines before flagging an error (default: 30)

    Returns:
        True if excessive duplicates are found (error detected), False otherwise
    """
    if not transcript_file.exists():
        logger.warning(f"File does not exist: {transcript_file}")
        return False

    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            logger.debug(f"Empty file: {transcript_file.name}")
            return False

        # Count occurrences of each content line (skip timestamp header)
        content_counts = {}

        # Skip lines with music indicators, Unicode characters, dots, or common filler words
        re_music = re.compile(r'music|yeah|all right|[^\x00-\x7F]', re.IGNORECASE)

        for line in lines:
            # Skip the timestamp header (first 32 characters)
            # Format: [HH:MM:SS.mmm --> HH:MM:SS.mmm]
            if re_music.search(line):
                continue

            if len(line) > 32:
                content = line[32:].strip()
            else:
                content = line.strip()

            if content:  # Only count non-empty lines
                if content not in content_counts:
                    content_counts[content] = 1
                else:
                    content_counts[content] += 1
                    if content_counts[content] > max_duplicate_lines:
                        logger.warning(
                            f"STUTTER ERROR DETECTED - {transcript_file.name}: "
                            f"*** {content} *** duplicate content lines found (threshold: {max_duplicate_lines}, count = {content_counts[content]})"
                        )
                        return True
    except Exception as exc:
        logger.error(f"Error reading {transcript_file}: {exc}")
        return False

    return False


def ai_summarize_episode(transcript_file: pathlib.Path) -> bool:
    """
    Generate an AI summary of a podcast episode transcript using an OpenAI-compatible API.

    Reads the transcript file and sends it to the configured LLM server to generate
    a comprehensive summary including show title, number, date, guests, songs, and
    album reviews if present.

    Args:
        transcript_file: Path to the transcript text file

    Returns:
        True if summary was generated successfully, False otherwise
    """
    if not transcript_file.exists():
        logger.error(f"Transcript file not found: {transcript_file}")
        return False

    summary_file = transcript_file.parent / "ai_show_summary.md"

    # Check if summary already exists
    if summary_file.exists() and summary_file.stat().st_size > 0:
        logger.info(f"SKIP - AI summary already exists: {summary_file.name}")
        return True

    logger.info(f"Generating AI summary for: {transcript_file.name}")

    try:
        # Read the transcript
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_content = f.read()

        if not transcript_content.strip():
            logger.error(f"Transcript file is empty: {transcript_file.name}")
            return False

        # Prepare the prompt
        system_prompt = (
            "You are an expert podcast analyzer. Generate comprehensive, well-structured "
            "summaries of podcast episodes that capture all key information, in Markdown format. "
            "Focus on clarity and detail to provide value to listeners seeking episode insights. "
        )

        user_prompt = (
            "Generate a show summary for this podcast. Highlight the show title, number, "
            "and date, as well as any special guests, and songs played on the episode. "
            "If the show includes an album review, clearly indicate the album name and "
            "the scoring for each song and overall rating.\n\n"
            f"Transcript:\n{transcript_content}"
        )

        # Prepare the API request
        headers = {
            "Content-Type": "application/json",
        }

        if OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": OPENAI_MAX_TOKENS,
            "temperature": OPENAI_TEMPERATURE
        }

        logger.info(f"Sending request to LLM API at {OPENAI_API_URL}")

        # Make the API request
        response = requests.post(
            OPENAI_API_URL,
            headers=headers,
            json=payload,
            timeout=600  # 10 minute timeout for long transcripts
        )

        response.raise_for_status()

        # Parse the response
        response_data = response.json()

        # Extract the summary from the response
        # Handle both OpenAI and LM Studio response formats
        if "choices" in response_data and len(response_data["choices"]) > 0:
            summary = response_data["choices"][0]["message"]["content"]
        else:
            logger.error(f"Unexpected API response format: {response_data}")
            return False

        # Write the summary to file
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

        logger.info(f"SUCCESS - AI summary saved to: {summary_file.name}")
        return True

    except requests.exceptions.RequestException as exc:
        logger.error(f"FAILED - API request error for {transcript_file.name}: {exc}")
        return False
    except json.JSONDecodeError as exc:
        logger.error(f"FAILED - Invalid JSON response for {transcript_file.name}: {exc}")
        return False
    except Exception as exc:
        logger.exception(f"ERROR - Failed to generate AI summary for {transcript_file.name}: {exc}")
        return False



def transcribe_file(media_file: pathlib.Path) -> bool:
    """
    Transcribe a media file using whisper-cli.exe.
    
    Args:
        media_file: Path to the media file to transcribe
        
    Returns:
        True if transcription succeeded, False otherwise
    """
    output_file = media_file.with_suffix('.txt')
    temp_wav_file = None

    # Check if transcription already exists
    if transcription_exists(media_file):
        logger.info(f"SKIP - Transcription already exists: {output_file.name}")
        return True
    
    logger.info(f"Transcribing: {media_file.name}")

    if media_file.suffix in VIDEO_EXTENSIONS:
        temp_wav_file = media_file.parent / "temp_audio.wav"

        # .\ffmpeg.exe -i "T:\2007-01-03 - RnR Geek Video- American Heartbreak in Japan\rockandrollgeek-44156-01-03-2007.mp4" -ar 16000 -ac 1 -c:a pcm_s16le output.wav
        cmd = [
            str(FFMPEG_EXECUTABLE),
            "-i", str(media_file),
            "-ar",    "16000",
            "-ac",    "1",
            "-c:a",   "pcm_s16le",
            temp_wav_file
        ]

        try:
            logger.info(f"Extracting audio from media file {media_file} using ffmpeg...")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            if result.returncode != 0:
                logger.error(f"FAILED - ffmpeg returned error code {result.returncode} while extracting audio")
                if result.stderr:
                    logger.error(f"ffmpeg error output: {result.stderr}")
                return False
            else:
                logger.info(f"Audio extracted to temporary file: {temp_wav_file.name}")
                media_file = temp_wav_file  # Update media_file to point to the extracted audio
        except Exception as exc:
            logger.exception(f"ERROR - Failed to extract audio from video {media_file.name}: {exc}")
            if temp_wav_file and temp_wav_file.exists():
                temp_wav_file.unlink()
            return False

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
            # Detect AI transcription errors by looking for duplicate lines
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                d = {}
                for line in lines:
                    content = line[32:].strip()
                    if content not in d:
                        d[content] = 1
                    else:
                        d[content] += 1

                # Detect transcription stutter error: more than 75 duplicate lines
                if detect_ai_transcription_stutter_error(output_file, max_duplicate_lines=75):
                    logger.error(f"FAILED - Detected excessive duplicate lines in transcription for {media_file.name}")
                    #output_file.unlink()  # Remove faulty transcription
                    #return False

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
    finally:
        if temp_wav_file and temp_wav_file.exists():
            temp_wav_file.unlink()


def main():
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
        
        # Get the transcript file path
        transcript_file = media_file.with_suffix('.txt')

        # Check if already transcribed
        if transcription_exists(media_file):
            skip_count += 1
            logger.info(f"SKIP - Transcription already exists")
            # Even if transcription exists, try to generate AI summary if missing
            ai_summarize_episode(transcript_file)
            continue
        
        # Transcribe the file
        if transcribe_file(media_file):
            success_count += 1
            # Generate AI summary after successful transcription
            ai_summarize_episode(transcript_file)
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

