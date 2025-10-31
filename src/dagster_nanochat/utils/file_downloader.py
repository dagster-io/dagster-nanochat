import os
import time

import requests


def download_file(url: str, filepath: str) -> bool:
    """Download a file from a URL to a local path with retries.

    Args:
        url: The URL to download from
        filepath: The local file path to save to (accepts string or path-like)

    Returns:
        True if download succeeded, False otherwise
    """
    max_attempts = 5

    # Convert to plain string to avoid UPath protocol issues
    filepath_str = str(filepath)
    temp_path_str = filepath_str + ".tmp"

    # Ensure parent directory exists
    parent_dir = os.path.dirname(filepath_str)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Write to temporary file first
            with open(temp_path_str, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):  # 1MB chunks
                    if chunk:
                        f.write(chunk)

            # Move temp file to final location
            os.replace(temp_path_str, filepath_str)
            return True

        except (requests.RequestException, IOError) as e:
            # Clean up any partial files
            for path in [temp_path_str, filepath_str]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2**attempt
                print(f"Download attempt {attempt} failed: {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Download failed after {max_attempts} attempts")
                return False

    return False
