import os
from typing import Optional, List, Tuple

import gradio

import facefusion.globals
from facefusion import wording
from facefusion.common_helper import get_first
from facefusion.filesystem import TEMP_DIRECTORY_PATH, clear_temp
from facefusion.filesystem import has_audio, has_image, filter_audio_paths, filter_image_paths
from facefusion.uis.core import register_ui_component
from facefusion.uis.typing import File

SOURCE_FILE: Optional[gradio.File] = None
SOURCE_FILE_2: Optional[gradio.File] = None
SOURCE_AUDIO: Optional[gradio.Audio] = None
SOURCE_IMAGE: Optional[gradio.Image] = None
SOURCE_IMAGE_2: Optional[gradio.Image] = None


def render() -> None:
    global SOURCE_FILE
    global SOURCE_FILE_2
    global SOURCE_AUDIO
    global SOURCE_IMAGE
    global SOURCE_IMAGE_2

    has_source_audio = has_audio(facefusion.globals.source_paths)
    has_source_image = has_image(facefusion.globals.source_paths)
    SOURCE_FILE = gradio.File(
        file_count='multiple',
        file_types=
        [
            '.mp3',
            '.wav',
            '.png',
            '.jpg',
            '.webp'
        ],
        label="Source 1",
        value=facefusion.globals.source_paths if has_source_audio or has_source_image else None,
        elem_id='source_file',
        elem_classes=['source_file']
    )
    SOURCE_FILE_2 = gradio.File(
        file_count='multiple',
        file_types=
        [
            '.png',
            '.jpg',
            '.webp'
        ],
        label="Source 2",
        value=facefusion.globals.source_paths if has_source_audio or has_source_image else None,
        elem_id='source_file_2',
        elem_classes=['source_file']
    )
    source_file_names = [source_file_value['name'] for source_file_value in
                         SOURCE_FILE.value] if SOURCE_FILE.value else None
    source_file_names_2 = [source_file_value['name'] for source_file_value in
                           SOURCE_FILE_2.value] if SOURCE_FILE_2.value else None
    source_audio_path = get_first(filter_audio_paths(source_file_names))
    source_image_path = get_first(filter_image_paths(source_file_names))
    source_image_path_2 = get_first(filter_image_paths(source_file_names_2))
    SOURCE_AUDIO = gradio.Audio(
        value=source_audio_path if has_source_audio else None,
        visible=has_source_audio,
        show_label=False,
        elem_id='source_audio'
    )
    with gradio.Row():
        SOURCE_IMAGE = gradio.Image(
            value=source_image_path if has_source_image else None,
            visible=has_source_image,
            show_label=False,
            elem_id='source_image',
            elem_classes=['source_image']
        )
        SOURCE_IMAGE_2 = gradio.Image(
            value=source_image_path_2 if has_source_image else None,
            visible=has_source_image,
            show_label=False,
            elem_id='source_image_2',
            elem_classes=['source_image']
        )
    register_ui_component('source_file', SOURCE_FILE)
    register_ui_component('source_file_2', SOURCE_FILE_2)
    register_ui_component('source_audio', SOURCE_AUDIO)
    register_ui_component('source_image', SOURCE_IMAGE)
    register_ui_component('source_image_2', SOURCE_IMAGE_2)


def listen() -> None:
    SOURCE_FILE.change(update, inputs=SOURCE_FILE, outputs=[SOURCE_AUDIO, SOURCE_IMAGE])
    SOURCE_FILE_2.change(update_2, inputs=SOURCE_FILE_2, outputs=[SOURCE_IMAGE_2])


def update(files: List[File]) -> Tuple[gradio.Audio, gradio.Image]:
    file_names = [file.name for file in files] if files else None
    temp_dir = TEMP_DIRECTORY_PATH
    os.makedirs(temp_dir, exist_ok=True)
    has_source_audio = has_audio(file_names)
    has_source_image = has_image(file_names)
    if has_source_audio or has_source_image:
        source_audio_path = get_first(filter_audio_paths(file_names))
        source_image_path = get_first(filter_image_paths(file_names))
        facefusion.globals.source_paths = file_names
        return gradio.update(value=source_audio_path, visible=has_source_audio), gradio.update(value=source_image_path,
                                                                                               visible=has_source_image)
    facefusion.globals.source_paths = None
    return gradio.update(value=None, visible=False), gradio.update(value=None, visible=False)


def update_2(files: List[File]) -> Tuple[gradio.Audio, gradio.Image]:
    file_names = [file.name for file in files] if files else None
    temp_dir = TEMP_DIRECTORY_PATH
    os.makedirs(temp_dir, exist_ok=True)
    has_source_image = has_image(file_names)
    if has_source_image:
        source_image_path = get_first(filter_image_paths(file_names))
        facefusion.globals.source_paths_2 = file_names
        return gradio.update(value=source_image_path, visible=has_source_image)
    facefusion.globals.source_paths_2 = None
    return gradio.update(value=None, visible=False)
