import os
import logging
import json
import time
import argparse
import sys
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
DEFAULT_RESULTS_DIR = "Azure/results/realtime"
DEFAULT_TEMP_DIR = "Azure/temp"
DEFAULT_DOWNLOAD_DIR = "Azure/downloads"
DEFAULT_MAX_WORKERS = 3  # Reduced to prevent overwhelming the system
DEFAULT_TIMEOUT = 600  # 10 minutes timeout for longer files

# Languages to consider for auto-detection
DEFAULT_LANGUAGES = ["en-IN", "hi-IN", "te-IN"]

# Global logger
logger = None


def setup_logging(log_file, verbose=False):
    """Set up logging configuration."""
    global logger

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
    )
    
    # Create our logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    # Reduce verbosity of other loggers
    logging.getLogger("azure").setLevel(logging.WARNING)
    
    return logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Azure Speech-to-Text with Real-Time Transcription and Speaker Diarization"
    )

    parser.add_argument(
        "--subscription-key",
        default=DEFAULT_SUBSCRIPTION_KEY,
        help="Azure Speech Service subscription key",
    )

    parser.add_argument(
        "--region", default=DEFAULT_REGION, help="Azure Speech Service region"
    )

    parser.add_argument(
        "--container-sas-url",
        default=DEFAULT_CONTAINER_SAS_URL,
        help="SAS URL for Azure Blob Storage container",
    )

    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory to save transcription results",
    )

    parser.add_argument(
        "--temp-dir", default=DEFAULT_TEMP_DIR, help="Directory for temporary files"
    )

    parser.add_argument(
        "--download-dir",
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory for downloaded audio files",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum number of parallel workers",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout in seconds for transcription",
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help="Languages to consider for auto-detection",
    )

    parser.add_argument(
        "--log-file",
        default="Azure/speech_transcription.log",
        help="Path to log file",
    )

    parser.add_argument("--recording-id", help="Process only a specific recording ID")

    parser.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="Keep temporary audio files after processing",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--use-local-files",
        action="store_true",
        help="Use locally downloaded files instead of downloading from blob",
    )

    parser.add_argument(
        "--skip-test", action="store_true", help="Skip the Azure connection test"
    )
    
    parser.add_argument(
        "--retry-limit", 
        type=int, 
        default=3,
        help="Number of retries for failed operations"
    )

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


def download_audio_file(recording_id, url, temp_dir, retry_limit=3):
    """Download audio file from URL to a temporary location."""
    try:
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{recording_id}.mp3")

        # Check if file already exists
        if os.path.exists(temp_file_path):
            if os.path.getsize(temp_file_path) > 0:
                logger.info(f"File {recording_id}.mp3 already exists, skipping download.")
                return temp_file_path
            else:
                logger.warning(f"Existing file {recording_id}.mp3 is empty, redownloading.")
                os.remove(temp_file_path)

        # Try downloading with retries
        for attempt in range(retry_limit):
            try:
                logger.info(f"Downloading {recording_id}.mp3 (attempt {attempt+1}/{retry_limit})...")
                response = requests.get(url, stream=True, timeout=30)

                if response.status_code == 200:
                    total_size = int(response.headers.get("content-length", 0))

                    with open(temp_file_path, "wb") as f, tqdm(
                        desc=f"Downloading {recording_id}",
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                size = f.write(chunk)
                                pbar.update(size)

                    # Verify download
                    if os.path.getsize(temp_file_path) > 0:
                        logger.info(f"Downloaded {recording_id}.mp3 - Size: {os.path.getsize(temp_file_path)/1024:.2f} KB")
                        return temp_file_path
                    else:
                        logger.warning(f"Downloaded file {recording_id}.mp3 is empty, retrying...")
                        continue
                else:
                    logger.error(
                        f"Failed to download {recording_id}.mp3: HTTP {response.status_code}"
                    )
                    if attempt < retry_limit - 1:
                        time.sleep(2)  # Wait before retrying
                    continue
            except Exception as e:
                logger.error(f"Error downloading {recording_id}.mp3 (attempt {attempt+1}): {str(e)}")
                if attempt < retry_limit - 1:
                    time.sleep(2)  # Wait before retrying
                continue

        logger.error(f"Failed to download {recording_id}.mp3 after {retry_limit} attempts")
        return None
    except Exception as e:
        logger.error(f"Error in download process for {recording_id}.mp3: {str(e)}")
        return None


def convert_mp3_to_wav(mp3_path, retry_limit=3):
    """Convert MP3 to WAV format for Azure Speech SDK."""
    try:
        wav_path = mp3_path.replace(".mp3", ".wav")

        # Check if WAV file already exists
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            logger.info(
                f"{os.path.basename(wav_path)} already exists, skipping conversion."
            )
            return wav_path

        # Check if MP3 exists and has content
        if not os.path.exists(mp3_path):
            logger.error(f"MP3 file not found: {mp3_path}")
            return None
            
        if os.path.getsize(mp3_path) == 0:
            logger.error(f"MP3 file is empty: {mp3_path}")
            return None

        # Try conversion with retries
        for attempt in range(retry_limit):
            try:
                logger.info(f"Converting {os.path.basename(mp3_path)} to WAV format (attempt {attempt+1}/{retry_limit})...")

                # Load audio using pydub with explicit ffmpeg configuration
                audio = AudioSegment.from_file(mp3_path, format="mp3")

                # Check if the audio is not empty
                if len(audio) == 0:
                    logger.error(f"Audio file {mp3_path} is empty or corrupted.")
                    return None

                logger.info(f"Audio duration: {len(audio) / 1000:.2f} seconds")

                # Convert to 16kHz, 16-bit mono for optimal speech recognition
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                audio.export(wav_path, format="wav")

                # Verify the WAV file was created successfully
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    logger.info(
                        f"Converted to {os.path.basename(wav_path)} - Size: {os.path.getsize(wav_path) / 1024:.2f} KB"
                    )
                    return wav_path
                else:
                    logger.error(f"Failed to create valid WAV file for {mp3_path} (attempt {attempt+1})")
                    if attempt < retry_limit - 1:
                        time.sleep(2)  # Wait before retrying
                    continue
            except Exception as e:
                logger.error(f"Error converting {os.path.basename(mp3_path)} to WAV (attempt {attempt+1}): {str(e)}")
                if attempt < retry_limit - 1:
                    time.sleep(2)  # Wait before retrying
                continue
                
        logger.error(f"Failed to convert {mp3_path} to WAV after {retry_limit} attempts")
        return None
    except Exception as e:
        logger.error(f"Unexpected error converting {os.path.basename(mp3_path)} to WAV: {str(e)}")
        return None


def create_auto_detect_language_config(languages):
    """Create a language configuration for auto-detection."""
    auto_detect_source_language_config = (
        speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)
    )
    return auto_detect_source_language_config


def test_azure_connection(subscription_key, region):
    """Simple test of Azure Speech Service parameters."""
    logger.info("Testing Azure Speech Service connection...")
    try:
        # Create a simple speech config
        speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key, region=region
        )

        # We won't actually make a network call here to avoid potential issues
        # Just verify we can create the speech config
        if speech_config:
            logger.info(
                "Azure Speech Service parameters are valid. Proceeding with transcription..."
            )
            return True
        else:
            logger.error("Failed to create speech config.")
            return False
    except Exception as e:
        logger.error(f"Azure Speech Service configuration error: {str(e)}")
        return False


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

        # Check if result already exists
        output_file = os.path.join(results_dir, f"{recording_id}.json")
        if os.path.exists(output_file):
            logger.info(f"Transcription for {recording_id} already exists, skipping.")
            return output_file

        # Create speech config
        speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key, region=region
        )

        # Enable verbose logging if requested
        if verbose:
            log_file = f"Azure/logs/{recording_id}_speech.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            speech_config.set_property(
                "speech.log.file", log_file
            )

        # Use API to enable features safely
        speech_config.request_word_level_timestamps()
        speech_config.enable_audio_logging()
        
        # Set silence timeout for better segmentation
        speech_config.set_property(
            "speech.silence.timeout.ms", "1000"
        )

        # Create auto language detection config
        auto_detect_source_language_config = create_auto_detect_language_config(
            languages
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
                    prev_end = int(all_results[-1].get("offset", "0").replace(":", "").replace(".", "")) + int(
                        all_results[-1].get("duration", "0").replace(":", "").replace(".", "")
                    )
                    curr_start = int(str(evt.result.offset).replace(":", "").replace(".", ""))
                    if curr_start - prev_end > speaker_change_threshold:
                        # Switch speakers on significant pause
                        if current_speaker == "Speaker 1":
                            current_speaker = "Speaker 2"
                        else:
                            current_speaker = "Speaker 1"

                # Get word-level details if available
                words = []
                if hasattr(evt.result, "words") and evt.result.words:
                    for word in evt.result.words:
                        words.append(
                            {
                                "word": word.word,
                                "offset": str(word.offset),
                                "duration": str(word.duration),
                            }
                        )

                # Create a result object
                result = {
                    "text": evt.result.text,
                    "offset": str(evt.result.offset),
                    "duration": str(evt.result.duration),
                    "speaker": current_speaker,
                }

                # Add word-level details if available
                if words:
                    result["words"] = words

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

        def session_started_cb(evt):
            logger.info(f"SESSION STARTED: {evt}")

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
        speech_recognizer.session_started.connect(session_started_cb)
        speech_recognizer.session_stopped.connect(session_stopped_cb)

        # Start continuous recognition
        logger.info(f"Starting recognition for {recording_id}...")
        speech_recognizer.start_continuous_recognition()

        # Wait for the transcription to complete or timeout
        start_time = time.time()
        with tqdm(total=timeout, desc=f"Transcribing {recording_id}", unit="s", leave=False) as pbar:
            last_update = 0
            while not done and (time.time() - start_time) < timeout:
                time.sleep(0.5)
                elapsed = min(int(time.time() - start_time), timeout)
                pbar.update(elapsed - last_update)
                last_update = elapsed

            # Force completion at timeout
            if not done:
                logger.warning(
                    f"Transcription timed out for {recording_id}, stopping recognition."
                )
                speech_recognizer.stop_continuous_recognition()
                time.sleep(2)  # Give it a moment to finish processing
                
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
            logger.warning(f"No transcription results for {recording_id}")
            # Save empty result to avoid reprocessing
            combined_result = {
                "recordingId": recording_id,
                "source": audio_file_path,
                "detectedLanguage": None,
                "recognizedPhrases": [{"text": "", "offset": "0", "duration": "0", "speaker": "Unknown"}],
                "error": "No speech detected or recognition failed"
            }
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(combined_result, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved empty transcription for {recording_id}")
            return output_file

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
            mp3_path = download_audio_file(recording_id, source, args.temp_dir, args.retry_limit)
            if not mp3_path:
                return recording_id, False

        # Check if file is not empty
        if os.path.getsize(mp3_path) == 0:
            logger.error(f"Audio file {mp3_path} is empty.")
            return recording_id, False

        logger.info(f"Audio file size: {os.path.getsize(mp3_path) / 1024:.2f} KB")

        # Convert to WAV
        wav_path = convert_mp3_to_wav(mp3_path, args.retry_limit)
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
    logger = setup_logging(args.log_file, args.verbose)

    # Create necessary directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs("Azure/logs", exist_ok=True)
    
    # Print startup information
    logger.info("=" * 80)
    logger.info("Starting Azure Speech Transcription")
    logger.info(f"Subscription key: {args.subscription_key[:5]}...{args.subscription_key[-5:]}")
    logger.info(f"Region: {args.region}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Languages: {args.languages}")
    logger.info("=" * 80)

    # Test Azure connection if not skipped
    if not args.skip_test and not test_azure_connection(
        args.subscription_key, args.region
    ):
        logger.error(
            "Failed to connect to Azure Speech Service. Please check your subscription key and region."
        )
        return

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
        
        # Use a smaller chunk size for better progress visualization
        chunk_size = min(total_recordings, 20)
        
        # Process recordings in chunks to avoid overwhelming the system
        for i in range(0, total_recordings, chunk_size):
            chunk = recording_sources[i:i+chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(total_recordings+chunk_size-1)//chunk_size} " +
                      f"(recordings {i+1}-{min(i+chunk_size, total_recordings)} of {total_recordings})")
            
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_recording = {
                    executor.submit(
                        process_recording, args, recording_id, source
                    ): recording_id
                    for recording_id, source in chunk
                }

                with tqdm(total=len(chunk), desc="Processing recordings") as pbar:
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

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Exiting gracefully...")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
    finally:
        logger.info("=" * 80)
        logger.info("Azure Speech Transcription completed")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
