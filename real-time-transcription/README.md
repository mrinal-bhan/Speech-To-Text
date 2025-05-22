# Azure Real-Time Transcription

This contains scripts for processing and transcribing audio recordings using Azure Speech Services.

## Setup Environment Variables

All the scripts now use environment variables to access sensitive API keys and credentials. 
Create a `.env` file in the root directory with the following format:

```
# Azure Speech Service credentials
AZURE_SPEECH_SUBSCRIPTION_KEY=your_subscription_key_here
AZURE_SPEECH_REGION=your_region_here

# Azure Blob Storage SAS URL
AZURE_BLOB_CONTAINER_SAS_URL=your_blob_container_sas_url_here

# Exotel API credentials
EXOTEL_SID=your_exotel_sid_here
EXOTEL_API_KEY=your_exotel_api_key_here
EXOTEL_API_TOKEN=your_exotel_api_token_here
```

You can create this file manually or use the following command to create it (make sure to replace the placeholder values with your actual credentials):

```bash
cat > .env << EOL
# Azure Speech Service credentials
AZURE_SPEECH_SUBSCRIPTION_KEY=your_subscription_key_here
AZURE_SPEECH_REGION=centralindia

# Azure Blob Storage SAS URL
AZURE_BLOB_CONTAINER_SAS_URL=https://account-name.blob.core.windows.net/container-name?sas-token-here

# Exotel API credentials
EXOTEL_SID=your_exotel_sid_here
EXOTEL_API_KEY=your_exotel_api_key_here
EXOTEL_API_TOKEN=your_exotel_api_token_here
EOL
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Real-Time Transcription

The `azure_realtime_transcribe.py` script transcribes audio files in real-time with speaker diarization:

```bash
python azure_realtime_transcribe.py --use-local-files --download-dir="path/to/audio/files"
```

### Exotel Recording Downloader

The `exotel_recorder.py` script downloads recordings from Exotel:

```bash
python exotel_recorder.py
```

## Configuration Options

The scripts support various command-line arguments:

- `--subscription-key`: Azure Speech Service subscription key
- `--region`: Azure Speech Service region
- `--container-sas-url`: SAS URL for Azure Blob Storage container
- `--results-dir`: Directory to save transcription results
- `--temp-dir`: Directory for temporary files
- `--download-dir`: Directory for downloaded audio files
- `--max-workers`: Maximum number of parallel workers
- `--timeout`: Timeout in seconds for transcription
- `--languages`: Languages to consider for auto-detection
- `--log-file`: Path to log file
- `--recording-id`: Process only a specific recording ID
- `--keep-temp-files`: Keep temporary audio files after processing
- `--verbose`: Enable verbose logging
- `--use-local-files`: Use locally downloaded files instead of downloading from blob
- `--skip-test`: Skip the Azure connection test

## Directory Structure

```
Azure/
├── .env                      # Environment variables
├── results/                  # Transcription results
│   └── realtime/             # Real-time transcription results
├── temp/                     # Temporary files
├── downloads/                # Downloaded audio files
├── logs/                     # Log files
└── real-time-transcription/  # Transcription scripts
    ├── azure_realtime_transcribe.py
    ├── exotel_recorder.py
    ├── check_env.py
    └── README.md
```

## Features

- **Real-time transcription** with improved accuracy for multilingual content
- **Speaker diarization** to identify different speakers in the conversation
- **Word-level timestamps** for precise alignment with audio
- **Parallel processing** of multiple audio files
- **Robust error handling** with retries and proper logging
- **Progress monitoring** with detailed status updates
- **Environment variables** for secure credential management

## Requirements

- Python 3.7+
- Azure Speech Service subscription
- Required Python packages:
  - azure-cognitiveservices-speech
  - azure-storage-blob
  - pydub
  - requests
  - tqdm

## Setup

1. Install the required packages:
   ```
   pip install azure-cognitiveservices-speech azure-storage-blob pydub requests tqdm
   ```

2. Make sure you have ffmpeg installed for audio conversion:
   - On macOS: `brew install ffmpeg`
   - On Ubuntu/Debian: `apt-get install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

### Basic Usage

Run the transcription script:

```
python Azure/azure_realtime_transcribe.py
```

### Command-line Options

The script supports various command-line arguments for flexibility:

```
usage: azure_realtime_transcribe.py [-h] [--subscription-key SUBSCRIPTION_KEY] [--region REGION]
                                   [--container-sas-url CONTAINER_SAS_URL] [--csv-file CSV_FILE]
                                   [--results-dir RESULTS_DIR] [--temp-dir TEMP_DIR]
                                   [--max-workers MAX_WORKERS] [--timeout TIMEOUT]
                                   [--languages LANGUAGES [LANGUAGES ...]] [--log-file LOG_FILE]
                                   [--recording-id RECORDING_ID] [--keep-temp-files] [--verbose]
```

#### Key Options:

- `--subscription-key`: Your Azure Speech Service subscription key
- `--region`: Your Azure Speech Service region (default: centralindia)
- `--csv-file`: Path to CSV file with recording IDs and URLs
- `--results-dir`: Directory to save transcription results
- `--max-workers`: Number of parallel workers (default: 5)
- `--recording-id`: Process only a specific recording ID
- `--languages`: Languages to consider for auto-detection (default: en-IN hi-IN te-IN)
- `--verbose`: Enable verbose logging
- `--keep-temp-files`: Keep temporary audio files after processing

### Examples

1. Process all recordings with default settings:
   ```
   python Azure/azure_realtime_transcribe.py
   ```

2. Process a specific recording:
   ```
   python Azure/azure_realtime_transcribe.py --recording-id 31ae08445f47e0b3f881d0978dfb195c
   ```

3. Use custom settings:
   ```
   python Azure/azure_realtime_transcribe.py --max-workers 3 --timeout 600 --results-dir custom_results
   ```

4. Enable verbose logging:
   ```
   python Azure/azure_realtime_transcribe.py --verbose
   ```

## Input Data Format

The script can read recording information from:

1. **CSV File**: A CSV file with columns `Id` and `RecordingUrls`
2. **Azure Blob Storage**: MP3 files stored in a blob container

## Output Format

The output JSON files contain:

- `recordingId`: The ID of the recording
- `source`: Path to the source audio file
- `recognizedPhrases`: Array of recognized speech segments, each containing:
  - Text content in various formats (lexical, display, etc.)
  - Speaker identification
  - Timestamps
  - Word-level details
  - Detected language information

## Troubleshooting

- **Audio format issues**: The script converts MP3 to WAV, but ensure your audio files are valid
- **Timeout errors**: Adjust the `--timeout` value if transcription takes too long
- **Language detection issues**: Ensure the languages in the `--languages` list match your content
- **Authentication errors**: Verify your Azure Speech Service subscription key and region

## Comparison with Batch Transcription

Real-time transcription offers several advantages over batch transcription:

1. **Better handling of code-mixed speech**: Real-time mode uses streaming context to better handle multilingual content
2. **Improved speaker diarization**: More accurate attribution of speech to speakers
3. **Higher accuracy**: Generally more accurate for conversational and multilingual audio

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
