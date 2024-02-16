import json
import os
import re
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Tuple

import requests
import yt_dlp

from facefusion import wording
from facefusion.filesystem import is_file, TEMP_DIRECTORY_PATH
from facefusion.mytqdm import mytqdm


def find_media_urls(page_url) -> List[Tuple[str, str]]:
    response = requests.get(page_url)
    page_content = response.text
    matches = re.findall(r'flashvars_\d+\s*=\s*({.*?});', page_content, re.DOTALL)

    media_urls = []
    # If a match is found, parse it as JSON
    if matches:
        flashvars_json = json.loads(matches[0])
        print(f"Found flashvars JSON, parsing: {flashvars_json}")
        title = flashvars_json.get('video_title', None)
        qualities = flashvars_json.get('defaultQuality', [])
        quality = int(qualities[-1]) if qualities else -1
        definitions = flashvars_json.get('mediaDefinitions', [])
        print(f"Title, quality, definitions: {title}, {quality}, {definitions}")
        flashvars_output = None
        max_quality = -1
        for definition in definitions:
            def_quality = definition.get('quality', -1)
            if isinstance(def_quality, str):
                def_quality = int(def_quality)
                if def_quality > max_quality:
                    flashvars_output = definition.get('videoUrl', None)
                    max_quality = def_quality
        if flashvars_output is not None:
            media_urls.append((flashvars_output, title))
            print(f"Found media URL: {media_urls[-1]}")
            return media_urls

    # Find other JSON matches
    hls_match = re.search(r"html5player\.setVideoHLS\('([^']+)'\);", page_content)
    title_match = re.search(r"html5player\.setVideoTitle\('([^']+)'\);", page_content)
    print(f"HLS Match: {hls_match}")
    if hls_match is not None and title_match is not None:
        hls_url = hls_match.group(1)
        title = title_match.group(1).strip()
        media_urls.append((hls_url, title))
        print(f"Found media URL: {media_urls[-1]}")
        return media_urls
    else:
        json_ld_pattern = re.compile(r'<script type="application/ld\+json">(.+?)</script>', re.DOTALL)
        matches_2 = json_ld_pattern.findall(page_content)
        if matches_2:
            for match in matches_2:
                json_data = json.loads(match)
                print(f"Found JSON: {json_data}")
                title = json_data.get('name', None)
                media_url = json_data.get('contentUrl', None)
                if media_url and title:
                    media_urls.append((media_url, title))
                    print(f"Found media URL: {media_urls[-1]}")
                    return media_urls
        else:
            print("No JSON matches found.")

    return list(set(media_urls))  # Remove duplicates by converting to a set and back to a list


def print_stream(stream):
    for line in stream:
        print(line, end='')


def download_convert_ts_to_mp4(ts_url, output_path):
    command = [
        'ffmpeg',
        '-i', ts_url,
        '-acodec', 'copy',
        '-vcodec', 'copy',
        '-movflags', '+faststart',
        '-f', 'mp4',
        output_path
    ]

    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        # Create threads to concurrently print stdout and stderr
        stdout_thread = threading.Thread(target=print_stream, args=(proc.stdout,))
        stderr_thread = threading.Thread(target=print_stream, args=(proc.stderr,))

        stdout_thread.start()
        stderr_thread.start()

        # Wait for the threads to finish
        stdout_thread.join()
        stderr_thread.join()

        # Check for 403 Forbidden error in stderr
        proc.stderr.seek(0)  # Rewind to the beginning of the stderr
        stdout_output = proc.stdout.read()
        stderr_output = proc.stderr.read()
        if 'HTTP error 403 Forbidden' in stderr_output or 'HTTP Error 403: Forbidden' in stdout_output:
            print(f"Failed to download and convert the video: {stderr_output}")
            return False

        if proc.returncode and proc.returncode != 0:
            print(f"Failed to download and convert the video: {stderr_output}")
            return False
        return True


def get_video_filename(title: str) -> str:
    # Replace spaces with underscores and remove disallowed characters for filenames
    safe_title = title.replace(" ", "_").translate({ord(i): None for i in r'\/:*?"<>|'})
    return f"{safe_title}.mp4"


def download_video(target_url: str) -> str:
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'skip_download': True,  # Initially, just get the info
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(target_url, download=False)
        video_title = info_dict.get('title')

    if video_title:
        video_filename = get_video_filename(video_title)
        video_path = os.path.join(TEMP_DIRECTORY_PATH, video_filename)

        if not os.path.exists(video_path):  # Download only if the file doesn't exist
            ydl_opts['skip_download'] = False  # Now, proceed to download
            ydl_opts['outtmpl'] = video_path
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([target_url])
                print(f"Downloaded video to {video_path}")
        else:
            print(f"Video already exists: {video_path}")

        return video_path
    else:
        print("Could not retrieve video title.")
        return ""


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    with ThreadPoolExecutor() as executor:
        for url in urls:
            executor.submit(get_download_size, url)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        total = get_download_size(url)
        if is_file(download_file_path):
            initial = os.path.getsize(download_file_path)
        else:
            initial = 0
        if initial < total:
            print("Downloading " + url + " to " + download_file_path)
            with mytqdm(total=total, initial=initial, desc=wording.get('downloading'), unit='B', unit_scale=True,
                        unit_divisor=1024) as progress:
                subprocess.Popen(
                    ['curl', '--create-dirs', '--silent', '--insecure', '--location', '--continue-at', '-', '--output',
                     download_file_path, url])
                current = initial
                while current < total:
                    if is_file(download_file_path):
                        current = os.path.getsize(download_file_path)
                        progress.update(current - progress.n)


@lru_cache(maxsize=None)
def get_download_size(url: str) -> int:
    try:
        response = urllib.request.urlopen(url, timeout=10)
        return int(response.getheader('Content-Length'))
    except (OSError, ValueError):
        return 0


def is_download_done(url: str, file_path: str) -> bool:
    if is_file(file_path):
        return get_download_size(url) == os.path.getsize(file_path)
    return False
