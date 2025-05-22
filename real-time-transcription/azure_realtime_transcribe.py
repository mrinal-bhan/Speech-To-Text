import os
import logging
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import ContainerClient
import requests
from pydub import AudioSegment
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default configuration
DEFAULT_SUBSCRIPTION_KEY = os.getenv("AZURE_SPEECH_SUBSCRIPTION_KEY")
DEFAULT_REGION = os.getenv("AZURE_SPEECH_REGION", "centralindia")
DEFAULT_CONTAINER_SAS_URL = os.getenv("AZURE_BLOB_CONTAINER_SAS_URL")
DEFAULT_RESULTS_DIR = "results/realtime"
DEFAULT_TEMP_DIR = "temp"
DEFAULT_DOWNLOAD_DIR = "downloads"
DEFAULT_MAX_WORKERS = 5
DEFAULT_TIMEOUT = 600
DEFAULT_LANGUAGES = ["en-IN", "hi-IN", "te-IN"]

# Global logger
logger = None


def setup_logging(log_file):
    """Set up logging configuration."""
    global logger
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    return logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Azure Speech-to-Text with Real-Time Transcription and Speaker Diarization"
    )
    parser.add_argument("--subscription-key", default=DEFAULT_SUBSCRIPTION_KEY)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--container-sas-url", default=DEFAULT_CONTAINER_SAS_URL)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--temp-dir", default=DEFAULT_TEMP_DIR)
    parser.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    parser.add_argument("--log-file", default="Azure/speech_transcription.log")
    parser.add_argument("--recording-id", help="Process only a specific recording ID")
    parser.add_argument("--keep-temp-files", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use-local-files", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def get_recordings_from_blob(container_sas_url, specific_id=None):
    """Get recording URLs from Azure Blob Storage."""
    logger.info("Fetching list of blobs from container...")
    try:
        container_client = ContainerClient.from_container_url(container_sas_url)
        recording_urls = []
        for blob in container_client.list_blobs():
            if blob.name.endswith(".mp3"):
                recording_id = os.path.splitext(blob.name)[0]
                if specific_id and recording_id != specific_id:
                    continue
                blob_url = f"{container_sas_url.split('?')[0]}/{blob.name}?{container_sas_url.split('?')[1]}"
                recording_urls.append((recording_id, blob_url))
        logger.info(f"Found {len(recording_urls)} audio files in blob storage.")
        return recording_urls
    except Exception as e:
        logger.error(f"Error fetching blobs: {str(e)}")
        return []


def get_local_recordings(download_dir, specific_id=None):
    """Get recordings from the local download directory."""
    logger.info(f"Looking for recordings in {download_dir}...")
    recordings = []
    try:
        if not os.path.exists(download_dir):
            logger.error(f"Download directory {download_dir} doesn't exist.")
            return []
        for filename in os.listdir(download_dir):
            if filename.endswith(".mp3"):
                recording_id = os.path.splitext(filename)[0]
                if specific_id and recording_id != specific_id:
                    continue
                file_path = os.path.join(download_dir, filename)
                recordings.append((recording_id, file_path))
        logger.info(f"Found {len(recordings)} local recordings in {download_dir}")
        return recordings
    except Exception as e:
        logger.error(f"Error getting local recordings: {str(e)}")
        return []


def download_audio_file(recording_id, url, temp_dir):
    """Download audio file from URL to a temporary location."""
    try:
        temp_file_path = os.path.join(temp_dir, f"{recording_id}.mp3")
        if os.path.exists(temp_file_path):
            logger.info(f"File {recording_id}.mp3 already exists, skipping download.")
            return temp_file_path

        logger.info(f"Downloading {recording_id}.mp3...")
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            with (
                open(temp_file_path, "wb") as f,
                tqdm(
                    desc=f"Downloading {recording_id}",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)
            logger.info(f"Downloaded {recording_id}.mp3")
            return temp_file_path
        else:
            logger.error(
                f"Failed to download {recording_id}.mp3: HTTP {response.status_code}"
            )
            return None
    except Exception as e:
        logger.error(f"Error downloading {recording_id}.mp3: {str(e)}")
        return None


def convert_mp3_to_wav(mp3_path):
    """Convert MP3 to WAV format for Azure Speech SDK."""
    try:
        wav_path = mp3_path.replace(".mp3", ".wav")

        if os.path.exists(wav_path):
            logger.info(
                f"{os.path.basename(wav_path)} already exists, skipping conversion."
            )
            return wav_path

        logger.info(f"Converting {os.path.basename(mp3_path)} to WAV format...")
        audio = AudioSegment.from_mp3(mp3_path)

        if len(audio) == 0:
            logger.error(f"Audio file {mp3_path} is empty or corrupted.")
            return None

        logger.info(f"Audio duration: {len(audio) / 1000:.2f} seconds")
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(wav_path, format="wav")

        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            logger.info(
                f"Converted to {os.path.basename(wav_path)} - Size: {os.path.getsize(wav_path) / 1024:.2f} KB"
            )
            return wav_path
        else:
            logger.error(f"Failed to create valid WAV file for {mp3_path}")
            return None
    except Exception as e:
        logger.error(f"Error converting {os.path.basename(mp3_path)} to WAV: {str(e)}")
        return None


def transcribe_audio_file(
    recording_id,
    audio_file_path,
    subscription_key,
    region,
    languages,
    timeout,
    results_dir,
    verbose=False,
):
    """Transcribe audio using Azure Speech SDK's real-time recognition with language detection."""
    try:
        logger.info(f"Starting transcription for {recording_id}...")

        output_file = os.path.join(results_dir, f"{recording_id}.json")
        if os.path.exists(output_file):
            logger.info(f"Transcription for {recording_id} already exists, skipping.")
            return output_file

        # Create speech config
        speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key, region=region
        )
        speech_config.request_word_level_timestamps()

        # Create auto language detection config
        auto_detect_source_language_config = (
            speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)
        )

        # Create audio config from the WAV file
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)

        # Create speech recognizer with auto language detection
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_detect_source_language_config,
        )

        # Initialize result container
        all_results = []
        detected_language = None
        done = False
        current_speaker = None
        speaker_change_threshold = (
            2000  # 2 seconds silence might indicate speaker change
        )

        # Define event handlers
        def recognized_cb(evt):
            nonlocal current_speaker, detected_language

            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Get the detected language if available
                if hasattr(evt.result, "properties") and evt.result.properties.get(
                    "LanguageDetectionResult"
                ):
                    new_lang = evt.result.properties["LanguageDetectionResult"]
                    if detected_language is None:
                        detected_language = new_lang
                        logger.info(f"Detected language: {detected_language}")

                # Simple speaker diarization based on pauses
                if not current_speaker:
                    current_speaker = "Speaker 1"
                elif len(all_results) > 0:
                    # Check time gap between current and previous utterance
                    prev_end = int(all_results[-1]["offset"]) + int(
                        all_results[-1]["duration"]
                    )
                    curr_start = int(evt.result.offset)
                    if curr_start - prev_end > speaker_change_threshold:
                        # Switch speakers on significant pause
                        if current_speaker == "Speaker 1":
                            current_speaker = "Speaker 2"
                        else:
                            current_speaker = "Speaker 1"

                # Create a result object
                result = {
                    "text": evt.result.text,
                    "offset": str(evt.result.offset),
                    "duration": str(evt.result.duration),
                    "speaker": current_speaker,
                }

                # Add language detection if available
                if hasattr(evt.result, "properties") and evt.result.properties.get(
                    "LanguageDetectionConfidence"
                ):
                    result["detected_language"] = {
                        "language": evt.result.properties["LanguageDetectionResult"],
                        "confidence": evt.result.properties[
                            "LanguageDetectionConfidence"
                        ],
                    }

                logger.info(f"Recognized ({current_speaker}): '{evt.result.text}'")
                all_results.append(result)

        def session_stopped_cb(evt):
            logger.info(f"SESSION STOPPED: {evt}")
            nonlocal done
            done = True

        def canceled_cb(evt):
            nonlocal done
            if evt.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Error details: {evt.error_details}")
            else:
                logger.warning(f"CANCELED: {evt.reason}")
            done = True

        # Connect callbacks to events
        speech_recognizer.recognized.connect(recognized_cb)
        speech_recognizer.canceled.connect(canceled_cb)
        speech_recognizer.session_stopped.connect(session_stopped_cb)
        speech_recognizer.session_started.connect(
            lambda evt: logger.info(f"SESSION STARTED: {evt}")
        )

        # Start continuous recognition
        logger.info(f"Starting recognition for {recording_id}...")
        speech_recognizer.start_continuous_recognition()

        # Wait for the transcription to complete or timeout
        start_time = time.time()
        with tqdm(total=timeout, desc=f"Transcribing {recording_id}", unit="s") as pbar:
            while not done and (time.time() - start_time) < timeout:
                time.sleep(1)
                elapsed = min(int(time.time() - start_time), timeout)
                pbar.update(elapsed - pbar.n)

            # Force completion at timeout
            if not done:
                logger.warning(
                    f"Transcription timed out for {recording_id}, stopping recognition."
                )
                speech_recognizer.stop_continuous_recognition()
                time.sleep(2)

        # Process and save results
        if all_results:
            # Combine results and format them
            combined_result = {
                "recordingId": recording_id,
                "source": audio_file_path,
                "detectedLanguage": detected_language,
                "recognizedPhrases": all_results,
            }

            # Save to JSON file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(combined_result, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Transcription saved for {recording_id} with {len(all_results)} segments"
            )
            return output_file
        else:
            logger.error(f"No transcription results for {recording_id}")
            return None

    except Exception as e:
        logger.error(f"Error transcribing {recording_id}: {str(e)}")
        return None


def process_recording(args, recording_id, source):
    """Process a single recording: download/copy, convert, and transcribe."""
    try:
        logger.info(f"Processing recording {recording_id}")

        # Check if result already exists
        result_path = os.path.join(args.results_dir, f"{recording_id}.json")
        if os.path.exists(result_path):
            logger.info(f"Transcription for {recording_id} already exists, skipping.")
            return recording_id, True

        # Handle the source file (either URL or local path)
        if args.use_local_files:
            # Source is already a local path
            mp3_path = source
            if not os.path.exists(mp3_path):
                logger.error(f"Local file not found: {mp3_path}")
                return recording_id, False

            # Copy to temp dir if needed
            temp_mp3_path = os.path.join(args.temp_dir, f"{recording_id}.mp3")
            if mp3_path != temp_mp3_path:
                import shutil

                os.makedirs(args.temp_dir, exist_ok=True)
                shutil.copy2(mp3_path, temp_mp3_path)
                logger.info(f"Copied {mp3_path} to {temp_mp3_path}")
                mp3_path = temp_mp3_path
        else:
            # Download from URL
            mp3_path = download_audio_file(recording_id, source, args.temp_dir)
            if not mp3_path:
                return recording_id, False

        # Check if file is not empty
        if os.path.getsize(mp3_path) == 0:
            logger.error(f"Audio file {mp3_path} is empty.")
            return recording_id, False

        logger.info(f"Audio file size: {os.path.getsize(mp3_path) / 1024:.2f} KB")

        # Convert to WAV
        wav_path = convert_mp3_to_wav(mp3_path)
        if not wav_path:
            return recording_id, False

        # Transcribe
        result_path = transcribe_audio_file(
            recording_id,
            wav_path,
            args.subscription_key,
            args.region,
            args.languages,
            args.timeout,
            args.results_dir,
            args.verbose,
        )
        success = result_path is not None

        # Clean up temporary files if not keeping them
        if not args.keep_temp_files and success:
            try:
                if not args.use_local_files or mp3_path != source:
                    os.remove(mp3_path)
                    logger.info(f"Removed temporary file: {mp3_path}")
                os.remove(wav_path)
                logger.info(f"Removed temporary file: {wav_path}")
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {str(e)}")

        return recording_id, success
    except Exception as e:
        logger.error(f"Error processing recording {recording_id}: {str(e)}")
        return recording_id, False


def main():
    """Main function to orchestrate the transcription pipeline."""
    # Parse command line arguments
    args = parse_arguments()

    # Set up logging
    global logger
    logger = setup_logging(args.log_file)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create necessary directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs("Azure/logs", exist_ok=True)

    try:
        # Get recordings
        if args.use_local_files:
            recording_sources = get_local_recordings(
                args.download_dir, args.recording_id
            )
            source_type = "local files"
        else:
            recording_sources = get_recordings_from_blob(
                args.container_sas_url, args.recording_id
            )
            source_type = "blob storage"

        if not recording_sources:
            logger.error(f"No recordings found to process from {source_type}.")
            return

        # Process recordings in parallel
        results = {}
        total_recordings = len(recording_sources)
        logger.info(
            f"Starting processing of {total_recordings} recordings from {source_type}..."
        )

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_recording = {
                executor.submit(
                    process_recording, args, recording_id, source
                ): recording_id
                for recording_id, source in recording_sources
            }

            with tqdm(total=total_recordings, desc="Overall progress") as pbar:
                for future in as_completed(future_to_recording):
                    recording_id = future_to_recording[future]
                    try:
                        recording_id, success = future.result()
                        results[recording_id] = success
                    except Exception as e:
                        logger.error(f"Error processing {recording_id}: {str(e)}")
                        results[recording_id] = False
                    pbar.update(1)

        # Summarize results
        successful = sum(1 for success in results.values() if success)
        logger.info(
            f"Transcription complete. {successful} of {total_recordings} recordings processed successfully."
        )

        # Print list of successfully transcribed recording IDs
        if successful > 0:
            logger.info("Successfully transcribed recordings:")
            for recording_id, success in results.items():
                if success:
                    logger.info(f"- {recording_id}")

        # Print list of failed recordings
        if successful < total_recordings:
            logger.warning("Failed recordings:")
            for recording_id, success in results.items():
                if not success:
                    logger.warning(f"- {recording_id}")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")


if __name__ == "__main__":
    main()
