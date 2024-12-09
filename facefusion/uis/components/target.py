import os.path
import traceback
from typing import Tuple, Optional, List

import gradio

from facefusion import wording, state_manager
from facefusion.download import download_video
from facefusion.face_store import clear_reference_faces, clear_static_faces
from facefusion.ffmpeg import extract_audio_from_video
from facefusion.filesystem import is_image, is_video, is_url, TEMP_DIRECTORY_PATH, get_file_size
from facefusion.uis.components.face_selector import clear_selected_faces
from facefusion.uis.core import register_ui_component, get_ui_component
from facefusion.uis.typing import File, ComponentOptions
from facefusion.vision import normalize_frame_color, get_video_frame

FILE_SIZE_LIMIT = 512 * 1024 * 1024
TARGET_FILE: Optional[gradio.File] = None
TARGET_IMAGE: Optional[gradio.Image] = None
TARGET_VIDEO: Optional[gradio.Video] = None
TARGET_PATH: Optional[gradio.Text] = None
SYNC_VIDEO_LIP: Optional[gradio.Checkbox] = None
SOURCE_FILES: Optional[gradio.File] = None


def render() -> None:
    global TARGET_FILE
    global TARGET_IMAGE
    global TARGET_VIDEO
    global TARGET_PATH
    global SYNC_VIDEO_LIP
    target_path = state_manager.get_item('target_path')
    is_target_path = is_url(target_path)
    is_target_image = is_image(target_path)
    is_target_video = is_video(target_path)
    TARGET_PATH = gradio.Text(
        label="Target URL/Remote Path",
        value=target_path if is_target_path else None,
        elem_id='ff_target_path',
    )
    TARGET_FILE = gradio.File(
        label=wording.get('uis.target_file'),
        file_count='single',
        file_types=
        [
            'image',
            'video'
        ],
        value=state_manager.get_item('target_path') if is_target_image or is_target_video else None
    )
    target_image_options: ComponentOptions = \
        {
            'show_label': False,
            'visible': False
        }
    target_video_options: ComponentOptions = \
        {
            'show_label': False,
            'visible': False
        }
    if is_target_image:
        target_image_options['value'] = TARGET_FILE.value.get('path')
        target_image_options['visible'] = True
    if is_target_video:
        if get_file_size(state_manager.get_item('target_path')) > FILE_SIZE_LIMIT:
            preview_vision_frame = normalize_frame_color(get_video_frame(state_manager.get_item('target_path')))
            target_image_options['value'] = preview_vision_frame
            target_image_options['visible'] = True
        else:
            target_video_options['value'] = TARGET_FILE.value.get('path')
            target_video_options['visible'] = True
    TARGET_IMAGE = gradio.Image(**target_image_options)
    TARGET_VIDEO = gradio.Video(**target_video_options)
    SYNC_VIDEO_LIP = gradio.Checkbox(
        label="Sync Lips to Audio",
        value=state_manager.get_item("sync_video_lip") and is_target_video,
        visible=is_target_video and "lip_syncer" in state_manager.get_item("frame_processors"),
        elem_id='sync_video_lip'
    )
    register_ui_component('target_image', TARGET_IMAGE)
    register_ui_component('target_video', TARGET_VIDEO)
    register_ui_component('target_file', TARGET_FILE)
    register_ui_component('sync_video_lip', SYNC_VIDEO_LIP)


def listen() -> None:
    TARGET_PATH.input(update_from_path, inputs=TARGET_PATH, outputs=[TARGET_PATH, TARGET_FILE])
    TARGET_FILE.change(update, inputs=TARGET_FILE, outputs=[TARGET_IMAGE, TARGET_VIDEO, TARGET_PATH, SYNC_VIDEO_LIP])
    global SOURCE_FILES
    SOURCE_FILES = get_ui_component('source_file')
    SYNC_VIDEO_LIP.change(update_sync_video_lip, inputs=[SYNC_VIDEO_LIP, SOURCE_FILES], outputs=[SOURCE_FILES])


def update_from_path(path: str) -> Tuple[gradio.update, gradio.update]:
    # TARGET_IMAGE, TARGET_VIDEO, TARGET_PATH, TARGET_FILE
    out_path = gradio.update(visible=True)
    out_file = gradio.update(visible=True)

    if not path:
        return out_path, out_file

    try:
        if is_url(path):
            print(f"Downloading video from {path}")
            downloaded_path = download_video(path)
            if not downloaded_path:
                raise ValueError("Failed to download video.")
            path = downloaded_path

        if is_image(path):
            state_manager.set_item('target_path', path)
            out_path = gradio.update(visible=False)
            out_file = gradio.update(value=path, visible=True)

        elif is_video(path):
            if get_file_size(path) > FILE_SIZE_LIMIT:
                raise ValueError("File size exceeds the limit.")
            state_manager.set_item('target_path', path)
            out_path = gradio.update(visible=False)
            out_file = gradio.update(value=path, visible=True)

        else:
            raise ValueError("Unsupported file type.")

    except Exception as e:
        print(f"Error processing path: {e}")
        state_manager.clear_item('target_path')
        out_file = gradio.update(value=None, visible=True)
        out_path = gradio.update(value=path, visible=True)
        path = None
    calling_method = traceback.extract_stack()[-2].name
    print(f"Calling method: {calling_method}")
    return out_path, out_file


def update(file: File) -> Tuple[gradio.update, gradio.update, gradio.update, gradio.update]:
    # Returns: TARGET_IMAGE, TARGET_VIDEO, TARGET_PATH, SYNC_VIDEO_LIP
    print("crf")
    clear_reference_faces()
    print("csf")
    clear_static_faces()
    print("cssf")
    clear_selected_faces()
    file_path = file.name if file else None

    if not file_path:
        state_manager.clear_item('target_path')
        return (gradio.update(value=None, visible=False),
                gradio.update(value=None, visible=False),
                gradio.update(visible=True),
                gradio.update(visible=False))

    try:
        if is_image(file_path):
            state_manager.set_item('target_path', file_path)
            return (gradio.update(value=file_path, visible=True),
                    gradio.update(value=None, visible=False),
                    gradio.update(visible=False),
                    gradio.update(visible=False))

        if is_video(file_path):
            print(f"Set tgt")
            state_manager.set_item('target_path', file_path)
            print("Returning")
            return (gradio.update(value=None, visible=False),
                    gradio.update(value=file_path, visible=True),
                    gradio.update(visible=False),
                    gradio.update(visible=True))

        raise ValueError("Unsupported file type.")

    except Exception as e:
        print(f"Error updating file: {e}")
        state_manager.clear_item('target_path')
    print(f"Failed to update file: {file_path},returning")
    return (gradio.update(value=None, visible=False),
            gradio.update(value=None, visible=False),
            gradio.update(visible=True),
            gradio.update(visible=False))


def update_sync_video_lip(sync_video_lip: bool, files: List[File]) -> gradio.update:
    state_manager.set_item("sync_video_lip", sync_video_lip)

    if sync_video_lip:
        target_video_path = state_manager.get_item('target_path')
        if not target_video_path or not is_video(target_video_path):
            print("No valid video path to sync.")
            return gradio.update()

        try:
            file_names = [file.name for file in files] if files else []
            target_video_extension = os.path.splitext(target_video_path)[1]
            audio_path = target_video_path.replace(target_video_extension, '.mp3')

            if not os.path.exists(audio_path):
                audio_path = extract_audio_from_video(target_video_path)

            if audio_path and audio_path not in file_names:
                file_names.append(audio_path)

            return gradio.update(value=file_names)

        except Exception as e:
            print(f"Error syncing video lip: {e}")
            return gradio.update()

    return gradio.update()
