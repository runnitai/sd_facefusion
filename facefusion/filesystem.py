import glob
import os
import shutil
from pathlib import Path
from typing import List, Optional

import filetype

import facefusion.globals
from modules.paths_internal import models_path, script_path

output_dir = os.path.join(script_path, 'outputs')
TEMP_DIRECTORY_PATH = os.path.join(output_dir, 'facefusion', 'temp')
TEMP_OUTPUT_VIDEO_NAME = 'temp.mp4'


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_frames_pattern = get_temp_frames_pattern(target_path, '*')
    return sorted(glob.glob(temp_frames_pattern))


def get_temp_frames_pattern(target_path: str, temp_frame_prefix: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, temp_frame_prefix + '.' + facefusion.globals.temp_frame_format)


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    return os.path.join(TEMP_DIRECTORY_PATH, target_name)


def get_temp_input_path(target_path: str) -> str:
    target_name = os.path.basename(target_path)
    if not os.path.exists(os.path.join(TEMP_DIRECTORY_PATH, 'input')):
        os.makedirs(os.path.join(TEMP_DIRECTORY_PATH, 'input'))
    return os.path.join(TEMP_DIRECTORY_PATH, 'input', target_name)


def get_temp_input_video_name(target_path: str) -> str:
    # Remove all the non-path characters from the target path
    target_path = ''.join([char for char in target_path if char.isalnum() or char in ['.', '_', '-', ' ']])
    # Replace spaces
    target_path = target_path.replace(' ', '_')
    target_path += '.mp4'
    return target_path


def get_temp_output_video_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_OUTPUT_VIDEO_NAME)


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_video_path = get_temp_output_video_path(target_path)
    if is_file(temp_output_video_path):
        if is_file(output_path):
            os.remove(output_path)
        shutil.move(temp_output_video_path, output_path)


def clear_temp() -> None:
    # Set max file size to 5GB
    max_file_size = 5 * 1024 * 1024 * 1024
    # Delete subfolders in TEMP_DIR_PATH
    for folder in glob.glob(os.path.join(TEMP_DIRECTORY_PATH, '**/*')):
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    # Get the size of all the files in TEMP_DIR_PATH
    total_temp_size = sum(os.path.getsize(file) for file in glob.glob(os.path.join(TEMP_DIRECTORY_PATH, '**/*')))
    # If the total size is greater than the max size, delete the oldest files until it's not
    if total_temp_size > max_file_size:
        # Get all the files in TEMP_DIR_PATH sorted by creation time
        files = sorted(glob.glob(os.path.join(TEMP_DIRECTORY_PATH, '**/*')), key=os.path.getctime)
        for file in files:
            # If the total size is less than the max size, stop deleting files
            if total_temp_size <= max_file_size:
                break
            file_size = os.path.getsize(file)
            # If the file is a directory, delete it
            if os.path.isdir(file):
                shutil.rmtree(file)
            # If the file is a file, delete it
            if os.path.isfile(file):
                os.remove(file)
            # Subtract the size of the deleted file from the total size
            total_temp_size -= file_size


def is_file(file_path: str) -> bool:
    return bool(file_path and os.path.isfile(file_path))


def is_directory(directory_path: str) -> bool:
    return bool(directory_path and os.path.isdir(directory_path))


def is_image(image_path: str) -> bool:
    if is_file(image_path):
        return filetype.helpers.is_image(image_path)
    return False


def are_images(image_paths: List[str]) -> bool:
    if image_paths:
        return all(is_image(image_path) for image_path in image_paths)
    return False


def is_video(video_path: str) -> bool:
    if is_file(video_path):
        return filetype.helpers.is_video(video_path)
    return False


def is_url(video_path: str) -> bool:
    return bool(video_path and video_path.startswith('http'))


def resolve_relative_path(path: str) -> str:
    model_name = os.path.basename(path)
    if model_name == "":
        return os.path.join(models_path, 'facefusion')
    return os.path.join(models_path, 'facefusion', model_name)


def list_module_names(path: str) -> Optional[List[str]]:
    if os.path.exists(path):
        files = os.listdir(path)
        return [Path(file).stem for file in files if not Path(file).stem.startswith(('.', '__'))]
    return None
