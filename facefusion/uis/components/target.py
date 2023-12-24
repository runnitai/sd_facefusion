import os.path
import shutil
import tempfile
from typing import Any, IO, Tuple, Optional

import gradio

import facefusion.globals
from facefusion import wording
from facefusion.download import download_video
from facefusion.face_store import clear_reference_faces, clear_static_faces
from facefusion.uis.core import register_ui_component
from facefusion.filesystem import is_image, is_video, is_url, TEMP_DIRECTORY_PATH

TARGET_PATH: Optional[gradio.Text] = None
TARGET_FILE: Optional[gradio.File] = None
TARGET_IMAGE: Optional[gradio.Image] = None
TARGET_VIDEO: Optional[gradio.Video] = None


def render() -> None:
    global TARGET_PATH
    global TARGET_FILE
    global TARGET_IMAGE
    global TARGET_VIDEO

    is_target_path = is_url(facefusion.globals.target_path)
    is_target_image = is_image(facefusion.globals.target_path)
    is_target_video = is_video(facefusion.globals.target_path)
    TARGET_PATH = gradio.Text(
        label="Target URL/Remote Path",
        value=facefusion.globals.target_path if is_target_path else None,
        elem_id='ff_target_path',
    )
    TARGET_FILE = gradio.File(
        label=wording.get('target_file_label'),
        file_count='single',
        file_types=
        [
            '.png',
            '.jpg',
            '.webp',
            '.mp4'
        ],
        value=facefusion.globals.target_path if is_target_image or is_target_video else None,
        elem_id='ff_target_file',
    )
    TARGET_IMAGE = gradio.Image(
        value=TARGET_FILE.value['name'] if is_target_image else None,
        visible=is_target_image,
        show_label=False
    )
    TARGET_VIDEO = gradio.Video(
        value=TARGET_FILE.value['name'] if is_target_video else None,
        visible=is_target_video,
        show_label=False
    )
    register_ui_component('target_image', TARGET_IMAGE)
    register_ui_component('target_video', TARGET_VIDEO)
    register_ui_component('target_file', TARGET_FILE)


def listen() -> None:
    TARGET_PATH.input(update_from_path, inputs=TARGET_PATH,
                      outputs=[TARGET_PATH, TARGET_FILE, TARGET_IMAGE, TARGET_VIDEO])
    TARGET_FILE.change(update, inputs=TARGET_FILE, outputs=[TARGET_PATH, TARGET_IMAGE, TARGET_VIDEO])


def update_from_path(path: str) -> Tuple[gradio.update, gradio.update, gradio.update, gradio.update]:
    if not path:
        return gradio.update(value=None, visible=True), gradio.update(value=None, visible=False), gradio.update(
            value=None, visible=False)
    if is_url(path):
        print(f"Downloading video from {path}")
        path = download_video(path)
        print(f"Downloaded video to {path}")
        if not path:
            return gradio.update(value=None, visible=True), gradio.update(value=None, visible=False), gradio.update(
                value=None, visible=False)

    root_path = facefusion.globals.restricted_path if facefusion.globals.restricted_path else "/mnt/private"
    absolute_path = os.path.abspath(path)
    # If the absolute path doesn't start with the root path or the TEMP_DIRECTORY_PATH, then it's not allowed
    if not absolute_path.startswith(root_path) and not absolute_path.startswith(TEMP_DIRECTORY_PATH):
        print(f"Path {path} is not allowed")
        return gradio.update(value=None, visible=True), gradio.update(value=path, visible=True), gradio.update(
            value=None, visible=False), gradio.update(value=None, visible=False)
    if is_image(path):
        facefusion.globals.target_path = path
        return gradio.update(value=path, visible=True), gradio.update(value=path, visible=True), gradio.update(
            value=path, visible=True), gradio.update(value=None, visible=False)
    if is_video(path):
        facefusion.globals.target_path = path
        return gradio.update(value=path, visible=True), gradio.update(value=path, visible=True), gradio.update(
            value=None, visible=False), gradio.update(value=path, visible=True)
    print(f"Invalid path {path}")
    facefusion.globals.target_path = None
    return gradio.update(value=None, visible=True), gradio.update(value=path, visible=True), gradio.update(value=None,
                                                                                                           visible=False), gradio.update(
        value=None, visible=False)


def update(file: IO[Any]) -> Tuple[gradio.Image, gradio.Video]:
    clear_reference_faces()
    clear_static_faces()
    # If file is a string, make it a file object
    if isinstance(file, str):
        file_path = file
    elif file:
        file_path = file.name
    else:
        file_path = None

    if file_path:
        temp_dir = TEMP_DIRECTORY_PATH
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, os.path.basename(file_path))
        if not os.path.exists(temp_path):
            shutil.copy(file_path, temp_path)
        try:
            os.remove(file_path)
        except:
            pass
        file_path = temp_path

    if file_path and is_image(file_path):
        facefusion.globals.target_path = file_path
        return gradio.update(value=file_path, visible=False), gradio.update(value=file_path,
                                                                            visible=True), gradio.update(value=None,
                                                                                                         visible=False)
    if file_path and is_video(file_path):
        facefusion.globals.target_path = file_path
        return gradio.update(value=file_path, visible=False), gradio.update(value=None, visible=False), gradio.update(
            value=file_path, visible=True)
    facefusion.globals.target_path = None
    return gradio.update(value=None, visible=True), gradio.update(value=None, visible=False), gradio.update(value=None,
                                                                                                            visible=False)
