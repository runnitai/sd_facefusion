import filetype
import glob
import os
import shutil
from pathlib import Path

import facefusion.globals
from typing import List, Optional

from modules.paths_internal import script_path

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


def clear_temp(some_file=None) -> None:
    src_files = []
    if facefusion.globals.source_paths:
        src_files = [f for f in facefusion.globals.source_paths if os.path.exists(f)]
    tgt_file = facefusion.globals.target_path
    for item in glob.glob(os.path.join(TEMP_DIRECTORY_PATH, '**/*')):
        if os.path.isdir(item):
            shutil.rmtree(item)
            continue
        if tgt_file and os.path.exists(tgt_file) and item == tgt_file:
            continue
        if item in src_files:
            continue
        if os.path.exists(item):
            os.remove(item)


def is_file(file_path: str) -> bool:
    return bool(file_path and os.path.isfile(file_path))


def is_directory(directory_path: str) -> bool:
    return bool(directory_path and os.path.isdir(directory_path))


def is_audio(audio_path: str) -> bool:
    return is_file(audio_path) and filetype.helpers.is_audio(audio_path)


def has_audio(audio_paths: List[str]) -> bool:
    if audio_paths:
        return any(is_audio(audio_path) for audio_path in audio_paths)
    return False


def is_image(image_path: str) -> bool:
    return is_file(image_path) and filetype.helpers.is_image(image_path)


def has_image(image_paths: List[str]) -> bool:
    if image_paths:
        return any(is_image(image_path) for image_path in image_paths)
    return False


def is_video(video_path: str) -> bool:
    return is_file(video_path) and filetype.helpers.is_video(video_path)


def filter_audio_paths(paths: List[str]) -> List[str]:
    if paths:
        return [path for path in paths if is_audio(path)]
    return []


def filter_image_paths(paths: List[str]) -> List[str]:
    if paths:
        return [path for path in paths if is_image(path)]
    return []


def resolve_relative_path(path: str) -> str:
    try:
        from modules.paths_internal import models_path
        return resolve_relative_path_auto(path)
    except:
        pass
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def list_directory(directory_path: str) -> Optional[List[str]]:
    if is_directory(directory_path):
        files = os.listdir(directory_path)
        return sorted([Path(file).stem for file in files if not Path(file).stem.startswith(('.', '__'))])
    return None


def is_url(video_path: str) -> bool:
    return bool(video_path and video_path.startswith('http'))


def resolve_relative_path_auto(path: str) -> str:
    from modules.paths_internal import models_path
    model_name = os.path.basename(path)
    if model_name == "" or path.endswith("models"):
        return os.path.join(models_path, 'facefusion')
    return os.path.join(models_path, 'facefusion', model_name)


def list_face_models() -> List[str]:
    from modules.paths_internal import models_path
    models = ['blendswap_256', 'ghost_unet_1_block', 'ghost_unet_2_block',
              'ghost_unet_3_block', 'inswapper_128', 'inswapper_128_fp16',
              'simswap_256', 'simswap_512_unofficial', 'uniface_256']
    model_dir = os.path.join(models_path, 'facefusion')
    return [model for model in models if os.path.exists(os.path.join(model_dir, f"{model}.onnx")) or "inswapper" not in model]
