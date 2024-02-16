import os.path
import shutil
from typing import Any, IO, Tuple, Optional, List

import gradio

import facefusion.globals
from facefusion import wording
from facefusion.download import download_video
from facefusion.face_store import clear_reference_faces, clear_static_faces
from facefusion.ffmpeg import extract_audio_from_video
from facefusion.filesystem import is_image, is_video, is_url, TEMP_DIRECTORY_PATH, clear_temp
from facefusion.uis.core import register_ui_component, get_ui_component
from facefusion.uis.components.source import update as source_update
from facefusion.uis.typing import File

TARGET_PATH: Optional[gradio.Text] = None
TARGET_FILE: Optional[gradio.File] = None
TARGET_IMAGE: Optional[gradio.Image] = None
TARGET_VIDEO: Optional[gradio.Video] = None
SYNC_VIDEO_LIP: Optional[gradio.Checkbox] = None
SOURCE_FILES: Optional[gradio.File] = None


def render() -> None:
    global TARGET_PATH
    global TARGET_FILE
    global TARGET_IMAGE
    global TARGET_VIDEO
    global SYNC_VIDEO_LIP

    is_target_path = is_url(facefusion.globals.target_path)
    is_target_image = is_image(facefusion.globals.target_path)
    is_target_video = is_video(facefusion.globals.target_path)
    TARGET_PATH = gradio.Text(
        label="Target URL/Remote Path",
        value=facefusion.globals.target_path if is_target_path else None,
        elem_id='ff_target_path',
    )
    TARGET_FILE = gradio.File(
        label = wording.get('uis.target_file'),
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
    SYNC_VIDEO_LIP = gradio.Checkbox(
        label="Sync Lips to Audio",
        value=facefusion.globals.sync_video_lip and is_target_video,
        visible=is_target_video and "lip_syncer" in facefusion.globals.frame_processors,
        elem_id='sync_video_lip'
    )
    register_ui_component('target_image', TARGET_IMAGE)
    register_ui_component('target_video', TARGET_VIDEO)
    register_ui_component('target_file', TARGET_FILE)
    register_ui_component('sync_video_lip', SYNC_VIDEO_LIP)


def listen() -> None:
    TARGET_PATH.input(update_from_path, inputs=TARGET_PATH,
                      outputs=[TARGET_PATH, TARGET_FILE, TARGET_IMAGE, TARGET_VIDEO])
    TARGET_FILE.change(update, inputs=TARGET_FILE, outputs=[TARGET_PATH, TARGET_IMAGE, TARGET_VIDEO, SYNC_VIDEO_LIP])
    source_audio = get_ui_component('source_audio')
    source_image = get_ui_component('source_image')
    global SOURCE_FILES
    SOURCE_FILES = get_ui_component('source_file')
    SYNC_VIDEO_LIP.change(update_sync_video_lip, inputs=[SYNC_VIDEO_LIP, SOURCE_FILES], outputs=[SOURCE_FILES])


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
        clear_temp()
        return gradio.update(value=path, visible=True), gradio.update(value=path, visible=True), gradio.update(
            value=path, visible=True), gradio.update(value=None, visible=False)
    if is_video(path):
        facefusion.globals.target_path = path
        clear_temp()
        return gradio.update(value=path, visible=True), gradio.update(value=path, visible=True), gradio.update(
            value=None, visible=False), gradio.update(value=path, visible=True)
    print(f"Invalid path {path}")
    facefusion.globals.target_path = None
    clear_temp()
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
        return (gradio.update(value=file_path, visible=False),
                gradio.update(value=file_path, visible=True),
                gradio.update(value=None, visible=False),
                gradio.update(visible=False))
    if file_path and is_video(file_path):
        facefusion.globals.target_path = file_path
        return (gradio.update(value=file_path, visible=False),
                gradio.update(value=None, visible=False),
                gradio.update(value=file_path, visible=True),
                gradio.update(visible=True))
    facefusion.globals.target_path = None
    return (gradio.update(value=None, visible=True),
            gradio.update(value=None, visible=False),
            gradio.update(value=None, visible=False),
            gradio.update(visible=False))


def update_sync_video_lip(sync_video_lip: bool, files: List[File]) -> None:
    facefusion.globals.sync_video_lip = sync_video_lip
    if sync_video_lip:
        target_video_path = facefusion.globals.target_path
        if target_video_path and is_video(target_video_path) and os.path.exists(target_video_path):
            file_names = [file.name for file in files] if files else []
            target_video_extension = os.path.splitext(target_video_path)[1]
            audio_path = target_video_path.replace(target_video_extension, '.mp3')
            if not os.path.exists(audio_path):
                audio_path = extract_audio_from_video(target_video_path)
            if audio_path not in file_names:
                file_names.append(audio_path)
            return gradio.update(value=file_names)
    return gradio.update()
