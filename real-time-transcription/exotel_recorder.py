import logging
import os
import csv
import requests
from urllib.parse import urlparse
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN")
DOWNLOAD_DIR = "Azure/exotel_recordings"
CSV_PATH = "/Users/mrinal/myEqual/Azure/exotel_recordings.csv"  # Must contain a 'RecordingUrls' column


def download_mp3(recording_url: str, save_dir: str) -> str:
    """
    Download an MP3 file from Exotel's API.

    Args:
        recording_url: The URL of the recording to download
        save_dir: Directory to save the downloaded file

    Returns:
        str: Path to the downloaded file, or None if download failed
    """
    try:
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.basename(urlparse(recording_url).path)
        if not filename.endswith(".mp3"):
            filename += ".mp3"

        local_path = os.path.join(save_dir, filename)

        if os.path.exists(local_path):
            logger.info(f"Already exists: {filename}")
            return local_path

        logger.info(f"Downloading {filename} ...")
        response = requests.get(
            recording_url,
            auth=HTTPBasicAuth(EXOTEL_API_KEY, EXOTEL_API_TOKEN),
            stream=True,
        )

        if response.status_code == 200:
            # Get total file size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            with (
                open(local_path, "wb") as f,
                tqdm(
                    desc=filename,
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

            logger.info(f"Downloaded: {filename}")
            return local_path
        else:
            logger.error(
                f"Failed to download {recording_url} - Status {response.status_code}"
            )
            return None

    except Exception as e:
        logger.error(f"Error downloading {recording_url}: {str(e)}")
        return None


def download_all_from_csv(csv_path: str, save_dir: str):
    """
    Download all recordings listed in the CSV file.

    Args:
        csv_path: Path to the CSV file containing recording URLs
        save_dir: Directory to save downloaded files
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            # Verify CSV has required column
            if "RecordingUrls" not in reader.fieldnames:
                logger.error("CSV file must contain a 'RecordingUrls' column")
                return

            # Count total rows for progress tracking
            total_recordings = sum(1 for _ in reader)
            f.seek(0)
            next(reader)  # Skip header row

            successful = 0
            failed = 0
            skipped = 0

            for row in tqdm(
                reader, total=total_recordings, desc="Processing recordings"
            ):
                recording_url = row.get("RecordingUrls")
                if not recording_url:
                    logger.warning("Missing recording_url in row")
                    skipped += 1
                    continue

                result = download_mp3(recording_url, save_dir)
                if result:
                    successful += 1
                else:
                    failed += 1

            logger.info("\nDownload Summary:")
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Skipped: {skipped}")
            logger.info(f"Total processed: {total_recordings}")

    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")


if __name__ == "__main__":
    download_all_from_csv(CSV_PATH, DOWNLOAD_DIR)
