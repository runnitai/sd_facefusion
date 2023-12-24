import json
import mimetypes
import os
import random
import re
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Tuple

import requests

from facefusion import wording
from facefusion.filesystem import is_file, get_temp_input_video_name, TEMP_DIRECTORY_PATH
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


def download_video(target_url: str) -> str:
    # If the target URL is from YouTube, use pytube to download it
    if 'youtube.com' in target_url or 'youtu.be' in target_url:
        print(f"Downloading video from {target_url} with youtube.")
        from pytube import YouTube
        youtube = YouTube(target_url)
        vid_name = youtube.title
        vid_path = get_temp_input_video_name(vid_name)
        # If vid_name is not a valid string, make up a random one
        if not vid_path or vid_path is True:
            vid_path = 'temp' + str(random.randint(0, 1000000)) + '.mp4'
        print(f"Downloading video to {vid_path} with name {vid_name}")
        youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(
            output_path=TEMP_DIRECTORY_PATH, filename=vid_path)
        return os.path.join(TEMP_DIRECTORY_PATH, vid_path)
    # Otherwise, try to download it with curl
    else:
        video_extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.wmv', '.flv', '.m4v', '.mpg', '.mpeg']
        # Do a request to get the content type
        print(f"Downloading video from {target_url} with curl.")
        request = urllib.request.Request(target_url, method='HEAD')
        response = urllib.request.urlopen(request)
        content_type = response.getheader('Content-Type')
        dest_file_name = response.getheader('Content-Disposition')
        # If the content type is a video, download it
        if content_type and content_type.startswith('video/'):
            video_extension = mimetypes.guess_extension(content_type)
            if video_extension in video_extensions:
                vid_name = dest_file_name.split('filename=')[1]
                vid_path = os.path.join(TEMP_DIRECTORY_PATH, vid_name)
                if os.path.exists(vid_path):
                    return vid_path
                urllib.request.urlretrieve(target_url, vid_path)
                return vid_path
        else:
            print("Checking for media URLS in page.")
            media_urls = find_media_urls(target_url)
            media_url, title = media_urls[0]
            title = title.replace('-', '').strip()
            title = title.replace(' ', '_')
            # Sanitize all non-path chars from the string and make sure it doesn't exceed path limits
            title = ''.join([char for char in title if char.isalnum() or char in ['.', '_', '-', ' ']])
            title = title[:250]
            if media_url is None:
                print("No media URL found.")
                return ""
            output_path = os.path.join(TEMP_DIRECTORY_PATH, title + '.mp4')
            if os.path.exists(output_path):
                return output_path
            if ".m3u8" in media_url or ".ts" in media_url:
                try:
                    # Download and convert the .ts file to .mp4
                    if download_convert_ts_to_mp4(media_url, output_path):
                        print(f"Downloaded and converted the video to {output_path}")
                        return output_path
                    else:
                        return ""
                except Exception as e:
                    print(f"Failed to download and convert the video: {e}")
            if ".mp4" in media_url:
                try:
                    urllib.request.urlretrieve(media_url, output_path)
                    print(f"Downloaded the video to {output_path}")
                    return output_path
                except Exception as e:
                    print(f"Failed to download the video: {e}")

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
