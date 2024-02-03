import os
import shutil
from typing import Optional, List

import gradio

import facefusion.globals
from facefusion import wording
from facefusion.filesystem import are_images, TEMP_DIRECTORY_PATH, clear_temp
from facefusion.uis.core import register_ui_component
from facefusion.uis.typing import File

SOURCE_FILE: Optional[gradio.File] = None
SOURCE_IMAGE: Optional[gradio.Image] = None


def render() -> None:
    global SOURCE_FILE
    global SOURCE_IMAGE

    are_source_images = are_images(facefusion.globals.source_paths)
    SOURCE_FILE = gradio.File(
        file_count='multiple',
        file_types=
        [
            '.png',
            '.jpg',
            '.webp'
        ],
        label=wording.get('source_file_label'),
        value=facefusion.globals.source_paths if are_source_images else None
    )
    source_file_names = [source_file_value['name'] for source_file_value in
                         SOURCE_FILE.value] if SOURCE_FILE.value else None
    SOURCE_IMAGE = gradio.Image(
        value=source_file_names[0] if are_source_images else None,
        visible=are_source_images,
        show_label=False
    )
    register_ui_component('source_image', SOURCE_IMAGE)


def listen() -> None:
    SOURCE_FILE.change(update, inputs=[SOURCE_FILE], outputs=[SOURCE_IMAGE])


def update(files: List[File]) -> gradio.Image:
    file_names = [file.name for file in files] if files else None
    largest_file_name = file_names[0] if file_names else None
    largest_file_size = 0
    temp_dir = TEMP_DIRECTORY_PATH
    os.makedirs(temp_dir, exist_ok=True)
    user_files = None
    if file_names is not None:
        user_files = []
        for file_name in file_names:
            file_path = os.path.join(temp_dir, os.path.basename(file_name))
            if not os.path.exists(file_path):
                shutil.copy(file_name, file_path)
            try:
                os.remove(file_name)
            except:
                pass
            file_size = os.path.getsize(file_path)
            if file_size > largest_file_size:
                largest_file_name = file_path
                largest_file_size = file_size
            user_files.append(file_path)
    if are_images(user_files):
        facefusion.globals.source_paths = user_files
        return gradio.update(value=largest_file_name, visible=True)
    facefusion.globals.source_paths = None
    clear_temp()
    return gradio.update(value=None, visible=False)
