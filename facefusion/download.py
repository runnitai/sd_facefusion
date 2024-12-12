import os
import shutil
import string
import subprocess
import urllib.request
from functools import lru_cache
from typing import List, Tuple
from urllib.parse import urlparse

import unicodedata
from tqdm import tqdm
from yt_dlp import YoutubeDL

from facefusion import wording, process_manager, state_manager, logger
from facefusion.filesystem import is_file, TEMP_DIRECTORY_PATH, get_file_size, remove_file
from facefusion.hash_helper import validate_hash
from facefusion.typing import DownloadSet


def get_video_filename(title: str) -> str:
    title = title.strip()
    # Split extension from the title
    title, ext = os.path.splitext(title)
    allowed_chars = set(string.ascii_letters + string.digits + '_-')
    # Replace spaces with underscores
    safe_title = title.replace(" ", "_")
    # Normalize unicode characters to ASCII equivalents
    safe_title = unicodedata.normalize('NFKD', safe_title)
    # Remove combining marks (accents)
    safe_title = ''.join(c for c in safe_title if not unicodedata.combining(c))
    # Remove disallowed characters
    safe_title = ''.join(c if c in allowed_chars else '_' for c in safe_title)
    # Remove leading and trailing dots or underscores
    safe_title = safe_title.strip('._')
    return safe_title


def download_video(target_url: str) -> str:
    try:
        # Step 1: Fetch video info
        ydl_opts_info = {
            'format': 'bestvideo+bestaudio/best',
            'skip_download': True  # Only fetch metadata
        }
        with YoutubeDL(ydl_opts_info) as ydl:
            info_dict = ydl.extract_info(target_url, download=False)
            video_title = info_dict.get('title')
            ext = info_dict.get('ext', 'mp4')  # Default extension to mp4 if missing

        if not video_title:
            print("Could not retrieve video title.")
            return ""

        # Step 2: Sanitize filename
        video_filename = get_video_filename(video_title)  # Use your provided function

        # Step 3: Check if the file already exists
        if os.path.exists(video_path):
            print(f"Video already exists: {video_path}")
            return video_path

        # Step 4: Set up download options with proper output path, forcing mp4 format
        ydl_opts_download = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': video_path,  # Save to sanitized path
        }
        with YoutubeDL(ydl_opts_download) as ydl:
            ydl.download([target_url])

        # Step 5: Verify the file exists
        if os.path.exists(video_path):
            print(f"Video downloaded successfully: {video_path}")
            return video_path
        else:
            print("Download completed, but the file could not be found.")
            return ""
    except Exception as e:
        print(f"An error occurred during video processing: {e}")
        return ""


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    for url in urls:
        download_file_name = os.path.basename(urlparse(url).path)
        download_file_path = os.path.join(download_directory_path, download_file_name)
        initial_size = get_file_size(download_file_path)
        download_size = get_download_size(url)

        if initial_size < download_size:
            with tqdm(total=download_size, initial=initial_size, desc=wording.get('downloading'), unit='B',
                      unit_scale=True, unit_divisor=1024, ascii=' =',
                      disable=state_manager.get_item('log_level') in ['warn', 'error']) as progress:
                subprocess.Popen(
                    [shutil.which('curl'), '--create-dirs', '--silent', '--insecure', '--location', '--continue-at',
                     '-', '--output', download_file_path, url])
                current_size = initial_size

                progress.set_postfix(file=download_file_name)
                while current_size < download_size:
                    if is_file(download_file_path):
                        current_size = get_file_size(download_file_path)
                        progress.update(current_size - progress.n)


@lru_cache(maxsize=None)
def get_download_size(url: str) -> int:
    try:
        response = urllib.request.urlopen(url, timeout=10)
        content_length = response.headers.get('Content-Length')
        return int(content_length)
    except (OSError, TypeError, ValueError):
        return 0


def is_download_done(url: str, file_path: str) -> bool:
    if is_file(file_path):
        return get_download_size(url) == get_file_size(file_path)
    return False


def conditional_download_hashes(download_directory_path: str, hashes: DownloadSet) -> bool:
    hash_paths = [hashes.get(hash_key).get('path') for hash_key in hashes.keys()]
    process_manager.check()
    if not state_manager.get_item('skip_download'):
        _, invalid_hash_paths = validate_hash_paths(hash_paths)
        if invalid_hash_paths:
            for index in hashes:
                if hashes.get(index).get('path') in invalid_hash_paths:
                    invalid_hash_url = hashes.get(index).get('url')
                    conditional_download(download_directory_path, [invalid_hash_url])
    valid_hash_paths, invalid_hash_paths = validate_hash_paths(hash_paths)
    for valid_hash_path in valid_hash_paths:
        valid_hash_file_name, _ = os.path.splitext(os.path.basename(valid_hash_path))
        logger.debug(wording.get('validating_hash_succeed').format(hash_file_name=valid_hash_file_name), __name__)
    for invalid_hash_path in invalid_hash_paths:
        invalid_hash_file_name, _ = os.path.splitext(os.path.basename(invalid_hash_path))
        print(f"Invalid hash path: {invalid_hash_path}")
        logger.error(wording.get('validating_hash_failed').format(hash_file_name=invalid_hash_file_name), __name__)
    if not invalid_hash_paths:
        process_manager.end()
    return not invalid_hash_paths


def conditional_download_sources(download_directory_path: str, sources: DownloadSet) -> bool:
    source_paths = [sources.get(source_key).get('path') for source_key in sources.keys()]
    process_manager.check()
    if not state_manager.get_item('skip_download'):
        _, invalid_source_paths = validate_source_paths(source_paths)
        if invalid_source_paths:
            for index in sources:
                if sources.get(index).get('path') in invalid_source_paths:
                    invalid_source_url = sources.get(index).get('url')
                    conditional_download(download_directory_path, [invalid_source_url])
    valid_source_paths, invalid_source_paths = validate_source_paths(source_paths)
    for valid_source_path in valid_source_paths:
        valid_source_file_name, _ = os.path.splitext(os.path.basename(valid_source_path))
        logger.debug(wording.get('validating_source_succeed').format(source_file_name=valid_source_file_name), __name__)
    for invalid_source_path in invalid_source_paths:
        invalid_source_file_name, _ = os.path.splitext(os.path.basename(invalid_source_path))
        print(f"Invalid source path: {invalid_source_path}")
        logger.error(wording.get('validating_source_failed').format(source_file_name=invalid_source_file_name),
                     __name__)
        if remove_file(invalid_source_path):
            logger.error(wording.get('deleting_corrupt_source').format(source_file_name=invalid_source_file_name),
                         __name__)
    if not invalid_source_paths:
        process_manager.end()
    return not invalid_source_paths


def validate_hash_paths(hash_paths: List[str]) -> Tuple[List[str], List[str]]:
    valid_hash_paths = []
    invalid_hash_paths = []

    for hash_path in hash_paths:
        if is_file(hash_path):
            valid_hash_paths.append(hash_path)
        else:
            invalid_hash_paths.append(hash_path)
    return valid_hash_paths, invalid_hash_paths


def validate_source_paths(source_paths: List[str]) -> Tuple[List[str], List[str]]:
    valid_source_paths = []
    invalid_source_paths = []

    for source_path in source_paths:
        if validate_hash(source_path):
            valid_source_paths.append(source_path)
        else:
            invalid_source_paths.append(source_path)
    return valid_source_paths, invalid_source_paths
