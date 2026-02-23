#!/usr/bin/env python3
"""
Audio File Monitor and Transcription System (S3)
Monitors S3 input bucket for MP3, M4A and MP4 files and transcribes them using WhisperX
For MP4 files, only the audio track is processed.
Uploads transcript artifacts to a separate S3 output bucket.
"""

import os
import time
import threading
import json
import re
import tempfile
from pathlib import Path
import whisperx
import gc
import torch
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioFileMonitor:
    def __init__(self):
        self.input_bucket = self.get_required_env("S3_INPUT_BUCKET")
        self.output_bucket = self.get_required_env("S3_OUTPUT_BUCKET")
        self.s3_client = boto3.client("s3")
        self.processing_lock = threading.Lock()
        self.current_language = None
        self.model_a = None
        self.metadata = None
        self.supported_extensions = {'.mp3', '.m4a', '.mp4'}
        self.setup_s3()
        self.setup_whisperx()

    @staticmethod
    def get_required_env(name):
        """Fetch required environment variable value."""
        value = os.getenv(name)
        if value is None or not value.strip():
            raise ValueError(f"Missing required environment variable: {name}")
        return value.strip()

    def setup_s3(self):
        """Verify S3 buckets are reachable."""
        logger.info("Validating S3 bucket access...")
        self.s3_client.head_bucket(Bucket=self.input_bucket)
        self.s3_client.head_bucket(Bucket=self.output_bucket)
        logger.info(f"✓ Input bucket: {self.input_bucket}")
        logger.info(f"✓ Output bucket: {self.output_bucket}")

    def detect_language_from_filename(self, filename):
        """Detect language from filename patterns"""
        filename_lower = filename.lower()
        if "-en.mp3" in filename_lower or "-en.m4a" in filename_lower or "-en.mp4" in filename_lower or "-en-" in filename_lower:
            return "en"
        return "fi"  # Default to Finnish

    def setup_whisperx(self):
        """Initialize WhisperX models"""
        logger.info("Initializing WhisperX models...")

        # Auto-detect device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"

        self.batch_size = 16

        logger.info(f"Device: {self.device} ({'CUDA detected' if self.device == 'cuda' else 'CUDA not available'})")
        logger.info(f"Compute type: {self.compute_type}")

        try:
            # Load Whisper model
            self.model = whisperx.load_model("large-v3", self.device, compute_type=self.compute_type)
            logger.info("✓ Whisper model loaded successfully")

            # Load diarization model
            hftoken = os.getenv("HF_TOKEN")
            self.diarize_model = whisperx.diarize.DiarizationPipeline(token=hftoken, device=self.device)
            logger.info("✓ Diarization model loaded")

            # Note: Alignment model will be loaded dynamically based on detected language

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def load_alignment_model(self, language_code):
        """Load alignment model for specific language"""
        if self.current_language != language_code:
            logger.info(f"Loading alignment model for language: {language_code}")
            try:
                self.model_a, self.metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
                self.current_language = language_code
                logger.info(f"✓ Alignment model loaded for {language_code}")
            except Exception as e:
                logger.error(f"Error loading alignment model for {language_code}: {e}")
                raise
        else:
            logger.info(f"Alignment model for {language_code} already loaded")

    def list_audio_objects(self):
        """List audio objects from input bucket."""
        paginator = self.s3_client.get_paginator("list_objects_v2")
        audio_objects = []

        for page in paginator.paginate(Bucket=self.input_bucket):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                if Path(key).suffix.lower() in self.supported_extensions:
                    audio_objects.append(obj)

        audio_objects.sort(key=lambda obj: obj["LastModified"])
        return audio_objects

    def build_markdown_output_key(self, object_key):
        """Build output key for markdown transcript at output bucket root."""
        return f"{Path(object_key).stem}.md"

    def upload_markdown_transcript(self, output_dir, object_key):
        """Upload only the markdown transcript file to output bucket root."""
        markdown_filename = self.build_markdown_output_key(object_key)
        markdown_path = Path(output_dir) / markdown_filename
        if not markdown_path.exists():
            raise FileNotFoundError(f"Expected markdown transcript not found: {markdown_path}")

        self.s3_client.upload_file(str(markdown_path), self.output_bucket, markdown_filename)
        logger.info(f"Uploaded markdown: s3://{self.output_bucket}/{markdown_filename}")

    def is_already_processed(self, object_key):
        """
        Check whether object has already been processed by looking for markdown
        transcript in the output bucket root.
        """
        markdown_key = self.build_markdown_output_key(object_key)
        try:
            self.s3_client.head_object(Bucket=self.output_bucket, Key=markdown_key)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def delete_s3_object(self, object_key):
        """Delete original audio file from S3 input bucket after successful processing."""
        try:
            self.s3_client.delete_object(Bucket=self.input_bucket, Key=object_key)
            logger.info(f"Deleted original audio file from S3: s3://{self.input_bucket}/{object_key}")
        except Exception as e:
            logger.error(f"Error deleting s3://{self.input_bucket}/{object_key}: {e}")

    def process_s3_object(self, object_key):
        """Process a single S3 audio object (MP3, M4A, or MP4)."""
        with self.processing_lock:
            try:
                object_path = Path(object_key)
                logger.info(f"Processing object: s3://{self.input_bucket}/{object_key}")

                filename_base = object_path.stem

                with tempfile.TemporaryDirectory(prefix="whisperx-s3-") as temp_dir:
                    local_root = Path(temp_dir)
                    target_dir = local_root / filename_base
                    target_dir.mkdir(parents=True, exist_ok=True)

                    local_audio_path = target_dir / object_path.name

                    logger.info(f"Downloading s3://{self.input_bucket}/{object_key}")
                    self.s3_client.download_file(self.input_bucket, object_key, str(local_audio_path))

                    # Detect language from filename
                    language_code = self.detect_language_from_filename(object_path.name)
                    logger.info(f"Detected language: {language_code}")

                    # Load alignment model for the detected language
                    self.load_alignment_model(language_code)

                    # Transcribe the file
                    self.transcribe_file(local_audio_path, target_dir)

                    # Upload only markdown transcript (no directory prefix)
                    self.upload_markdown_transcript(target_dir, object_key)
                    
                    # Delete the original audio file from S3 after successful processing and upload
                    self.delete_s3_object(object_key)

                return True

            except Exception as e:
                logger.error(f"Error processing s3://{self.input_bucket}/{object_key}: {e}")
                return False

    def transcribe_file(self, audio_path, output_dir):
        """Transcribe audio file using WhisperX pipeline"""
        start_time = time.time()

        logger.info(f"Starting transcription of {audio_path.name}")

        try:
            # Load audio
            logger.info("Loading audio...")
            audio = whisperx.load_audio(str(audio_path))
            audio_duration = len(audio) / 16000
            logger.info(f"Audio loaded: {audio_duration:.2f} seconds duration")

            # Transcribe with detected language
            logger.info(f"Transcribing audio (language: {self.current_language})...")
            transcribe_start = time.time()
            result = self.model.transcribe(audio, batch_size=self.batch_size, language=self.current_language)
            transcribe_time = time.time() - transcribe_start
            logger.info(f"Transcription completed in {transcribe_time:.2f}s ({len(result['segments'])} segments)")

            # Align
            logger.info("Aligning transcript...")
            align_start = time.time()
            result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, self.device, return_char_alignments=False)
            align_time = time.time() - align_start
            logger.info(f"Alignment completed in {align_time:.2f}s")

            # Diarize
            logger.info("Performing speaker diarization...")
            diarize_start = time.time()
            diarize_segments = self.diarize_model(audio)
            diarize_time = time.time() - diarize_start
            logger.info(f"Diarization completed in {diarize_time:.2f}s")

            # Assign speakers
            logger.info("Assigning speakers to words...")
            assign_start = time.time()
            result = whisperx.assign_word_speakers(diarize_segments, result)
            assign_time = time.time() - assign_start
            logger.info(f"Speaker assignment completed in {assign_time:.2f}s")

            # Save results
            self.save_transcript(result, audio_path, output_dir, audio_duration, start_time)

            # Cleanup
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            processing_time = time.time() - start_time
            speed_ratio = audio_duration / processing_time
            logger.info(f"✓ Transcription completed for {audio_path.name}")
            logger.info(f"Processing time: {processing_time:.2f}s (speed ratio: {speed_ratio:.2f}x)")

        except Exception as e:
            logger.error(f"Error during transcription of {audio_path.name}: {e}")
            raise

    def save_transcript(self, result, audio_path, output_dir, audio_duration, start_time):
        """Save transcript to file"""
        # Use base filename without extension for output files
        base_filename = audio_path.stem
        markdown_file = output_dir / f"{base_filename}.md"

        # Save .md format (new)
        self.save_markdown_transcript(result, audio_path, markdown_file, audio_duration, start_time)
        logger.info(f"Markdown transcript saved to {markdown_file}")

    def save_markdown_transcript(self, result, audio_path, markdown_file, audio_duration, start_time):
        """Save transcript in markdown format in chronological order"""
        processing_time = time.time() - start_time
        speed_ratio = audio_duration / processing_time

        with open(markdown_file, 'w', encoding='utf-8') as f:
            # Write header information
            f.write(f"# Diarized Transcript\n\n")
            f.write(f"**Audio file:** {audio_path.name}  \n")
            f.write(f"**Language:** {self.current_language}  \n")
            f.write(f"**Device:** {self.device}  \n")
            f.write(f"**Duration:** {audio_duration:.2f} seconds  \n")
            f.write(f"**Processing time:** {processing_time:.2f} seconds  \n")
            f.write(f"**Speed ratio:** {speed_ratio:.2f}x  \n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            f.write("---")

            # Write segments in chronological order
            current_speaker = None
            for i, segment in enumerate(result["segments"]):
                speaker = segment.get('speaker', 'UNKNOWN')
                start_time_seg = segment['start']
                end_time_seg = segment['end']
                text = segment['text'].strip()

                # Add speaker header when speaker changes
                if speaker != current_speaker:
                    start_time_str = time.strftime('%H:%M:%S', time.gmtime(start_time_seg))
                    f.write(f"\n\n## {speaker} ({start_time_str})\n\n")
                    current_speaker = speaker

                # Write the segment text
                if text:  # Only write non-empty text
                    if not text.endswith(" "):
                        text += " "
                    f.write(f"{text}")

    def save_json_transcript(self, result, audio_path, json_file, audio_duration, start_time):
        """Save transcript in JSON format"""
        processing_time = time.time() - start_time
        speed_ratio = audio_duration / processing_time

        # Prepare JSON data
        json_data = {
            "audio_file": audio_path.name,
            "language": self.current_language,
            "device": self.device,
            "duration": audio_duration,
            "processing_time": processing_time,
            "speed_ratio": speed_ratio,
            "segments": result["segments"],
            "generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

    def process_bucket_once(self):
        """Process currently available audio objects from S3 input bucket once."""
        logger.info(f"Running single-pass processing for S3 bucket: s3://{self.input_bucket}")
        logger.info("Language detection:")
        logger.info("  - Files with '-en.mp3', '-en.m4a', '-en.mp4' or '-en-' in filename: English transcription")
        logger.info("  - All other files: Finnish transcription (default)")
        logger.info(f"Output bucket: s3://{self.output_bucket}")
        logger.info("Supported formats: MP3, M4A, MP4")
        try:
            audio_objects = self.list_audio_objects()

            if audio_objects:
                logger.info(f"Found {len(audio_objects)} audio object(s) in input bucket")
            else:
                logger.info("No audio objects found in input bucket")
                return

            for obj in audio_objects:
                object_key = obj["Key"]
                if self.is_already_processed(object_key):
                    logger.info(f"Skipping already processed object: {object_key}")
                    continue
                if not self.is_recording(object_key, obj["LastModified"]):
                    logger.info(f"Object is ready (recording finished), processing: {object_key}")
                    self.process_s3_object(object_key)
                else:
                    logger.info(f"Object is still being recorded, skipping for now: {object_key}")
        except Exception as e:
            logger.error(f"Error during single-pass bucket processing: {e}")

    def is_recording(self, object_key, last_modified):
        """
        Check if an S3 object is likely still being recorded by comparing filename
        timestamp with object's last modified time.

        Only performs check for filenames in format: YYYYMMDD_HHMM-name.ext
        For files without this format, assumes not recording (returns False).
        """
        filename = Path(object_key).name

        # Pattern for filename: YYYYMMDD_HHMM-*.mp3 or *.m4a or *.mp4
        pattern = r'^(\d{8})_(\d{4})-.*\.(mp3|m4a|mp4)$'
        match = re.search(pattern, filename, re.IGNORECASE)

        if not match:
            # If filename doesn't match timestamp format, assume not recording
            # This allows immediate processing of files without timestamp format
            return False

        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMM

        try:
            # Parse filename date/time
            filename_datetime = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")

            # LastModified from S3 is timezone-aware UTC datetime
            modified_dt = last_modified.replace(tzinfo=None)

            # Compare date and time up to the minute
            return (
                filename_datetime.year == modified_dt.year
                and filename_datetime.month == modified_dt.month
                and filename_datetime.day == modified_dt.day
                and filename_datetime.hour == modified_dt.hour
                and filename_datetime.minute == modified_dt.minute
            )

        except ValueError as e:
            logger.warning(f"Could not parse date/time from object key {object_key}: {e}")
            return False

def main():
    """Main function to run one-pass S3 processing."""
    logger.info("=" * 50)
    logger.info("AUDIO FILE TRANSCRIPTION SYSTEM (S3 SINGLE RUN)")
    logger.info("=" * 50)

    monitor = AudioFileMonitor()
    monitor.process_bucket_once()

    logger.info("S3 single-pass run completed")

if __name__ == "__main__":
    main()
