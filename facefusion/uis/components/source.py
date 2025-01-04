import os
from typing import Optional, List, Tuple

import gradio

from facefusion import wording, state_manager
from facefusion.common_helper import get_first
from facefusion.filesystem import has_audio, has_image, filter_audio_paths, filter_image_paths
from facefusion.processors.classes.style_changer import StyleChanger
from facefusion.temp_helper import get_temp_directory_path
from facefusion.uis.core import register_ui_component, get_ui_components
from facefusion.uis.typing import File

SOURCE_FILE: Optional[gradio.File] = None
SOURCE_FILE_2: Optional[gradio.File] = None
SOURCE_AUDIO: Optional[gradio.Audio] = None
SOURCE_AUDIO_2: Optional[gradio.Audio] = None
SOURCE_IMAGE: Optional[gradio.Image] = None
SOURCE_IMAGE_2: Optional[gradio.Image] = None
AUDIO_ROW: Optional[gradio.Row] = None
IMAGE_ROW: Optional[gradio.Row] = None


def render() -> None:
    global SOURCE_FILE
    global SOURCE_FILE_2
    global SOURCE_AUDIO
    global SOURCE_AUDIO_2
    global SOURCE_IMAGE
    global SOURCE_IMAGE_2
    global AUDIO_ROW
    global IMAGE_ROW

    has_source_audio = has_audio(state_manager.get_item('source_paths'))
    has_source_image = has_image(state_manager.get_item('source_paths'))
    has_source_audio_2 = has_audio(state_manager.get_item('source_paths_2'))
    has_source_image_2 = has_image(state_manager.get_item('source_paths_2'))
    SOURCE_FILE = gradio.File(
        label=wording.get('uis.source_file'),
        file_count='multiple',
        file_types=
        [
            'audio',
            'image'
        ],
        value=state_manager.get_item('source_paths') if has_source_audio or has_source_image else None
    )

    SOURCE_FILE_2 = gradio.File(
        label=wording.get('uis.source_file_2'),
        file_count='multiple',
        file_types=
        [
            'audio',
            'image'
        ],
        value=state_manager.get_item('source_paths_2') if has_source_audio_2 or has_source_image_2 else None
    )
    source_file_names = [source_file_value['name'] for source_file_value in
                         SOURCE_FILE.value] if SOURCE_FILE.value else None
    source_file_names_2 = [source_file_value['name'] for source_file_value in
                           SOURCE_FILE_2.value] if SOURCE_FILE_2.value else None

    source_audio_path = get_first(filter_audio_paths(source_file_names))
    source_audio_path_2 = get_first(filter_audio_paths(source_file_names_2))

    source_image_path = get_first(filter_image_paths(source_file_names))
    source_image_path_2 = get_first(filter_image_paths(source_file_names_2))
    with gradio.Row(visible=has_source_audio) as AUDIO_ROW:
        SOURCE_AUDIO = gradio.Audio(
            value=source_audio_path if has_source_audio else None,
            visible=has_source_audio,
            show_label=False
        )
        SOURCE_AUDIO_2 = gradio.Audio(
            value=source_audio_path_2 if has_source_audio_2 else None,
            visible=has_source_audio_2,
            show_label=False
        )
    with gradio.Row(visible=has_source_image or has_source_image_2) as IMAGE_ROW:
        SOURCE_IMAGE = gradio.Image(
            value=source_image_path if has_source_image else None,
            visible=has_source_image,
            show_label=False,
            elem_id='source_image',
            elem_classes=['source_image']
        )
        SOURCE_IMAGE_2 = gradio.Image(
            value=source_image_path_2 if has_source_image else None,
            visible=has_source_image_2,
            show_label=False,
            elem_id='source_image_2',
            elem_classes=['source_image']
        )
    register_ui_component('source_file', SOURCE_FILE)
    register_ui_component('source_file_2', SOURCE_FILE_2)
    register_ui_component('source_audio', SOURCE_AUDIO)
    register_ui_component('source_audio_2', SOURCE_AUDIO_2)
    register_ui_component('source_image', SOURCE_IMAGE)
    register_ui_component('source_image_2', SOURCE_IMAGE_2)


def listen() -> None:
    SOURCE_FILE.change(update_1, inputs=SOURCE_FILE, outputs=[SOURCE_AUDIO, SOURCE_IMAGE, AUDIO_ROW, IMAGE_ROW])
    SOURCE_FILE_2.change(update_2, inputs=SOURCE_FILE_2, outputs=[SOURCE_AUDIO_2, SOURCE_IMAGE_2, AUDIO_ROW, IMAGE_ROW])
    SOURCE_FILE.clear(clear_1)
    SOURCE_FILE_2.clear(clear_2)

    for ui_component in get_ui_components(
            ['processors_checkbox_group', 'style_target_radio', 'style_changer_skip_head_checkbox',
             'style_changer_model_dropdown']):
        ui_component.change(remote_update, inputs=[SOURCE_FILE, SOURCE_FILE_2],
                            outputs=[SOURCE_AUDIO, SOURCE_IMAGE, SOURCE_AUDIO_2, SOURCE_IMAGE_2, AUDIO_ROW, IMAGE_ROW])


def clear_1() -> None:
    actual_clear(False)


def clear_2() -> None:
    actual_clear(True)


def update_1(files: List[File]) -> Tuple[gradio.Audio, gradio.update, gradio.update, gradio.update]:
    file_names = [file.name for file in files] if files else []
    return actual_update(file_names)


def update_2(files: List[File]) -> Tuple[gradio.update, gradio.update, gradio.update, gradio.update]:
    file_names = [file.name for file in files] if files else []
    return actual_update(file_names, True)


def actual_clear(is_src_2: bool = False) -> None:
    state_key = 'source_paths_2' if is_src_2 else 'source_paths'
    state_manager.clear_item(state_key)


def actual_update(file_names: List[str], is_src_2: bool = False) -> Tuple[
    gradio.update, gradio.update, gradio.update, gradio.update]:
    style_changer = StyleChanger()
    target = state_manager.get_item('style_changer_target')
    state_key = 'source_paths_2' if is_src_2 else 'source_paths'
    src_idx = 1 if is_src_2 else 0
    source_dict = state_manager.get_item('source_frame_dict')
    if not source_dict:
        print('source_dict is empty')
        source_dict = {}
    if 'source' in target and 'style_changer' in state_manager.get_item('processors'):
        all_image_files = filter_image_paths(file_names)
        for base_file in all_image_files:
            file, ext = os.path.splitext(base_file)
            styled_file = os.path.join(get_temp_directory_path(base_file), f'ff_styled{ext}')
            if os.path.exists(styled_file):
                os.remove(styled_file)
            styled_file = style_changer.process_src_image(base_file, styled_file)
            # Replace the original file name with the styled file name
            file_names[file_names.index(base_file)] = styled_file
    
    # Update state or clear it
    has_audio_files = has_audio(file_names)
    has_image_files = has_image(file_names)
    if file_names:
        source_dict[src_idx] = file_names
        state_manager.set_item(state_key, file_names)
        print(f"source_dict: {source_dict} src 2 {is_src_2}")
        state_manager.set_item('source_frame_dict', source_dict)
    else:
        source_dict[src_idx] = []
        state_manager.clear_item(state_key)
        print(f"source_dict: {source_dict} src 2 {is_src_2} (empty)")
        state_manager.set_item('source_frame_dict', source_dict)

    # Return UI updates
    return (
        gradio.update(value=get_first(filter_audio_paths(file_names)), visible=has_audio_files),
        gradio.update(value=get_first(filter_image_paths(file_names)), visible=has_image_files),
        gradio.update(visible=has_audio_files),
        gradio.update(visible=has_image_files)
    )


def remote_update(files1, files2) -> Tuple[
    gradio.update, gradio.update, gradio.update, gradio.update, gradio.update, gradio.update]:
    if not files1 and not files2:
        # If both file inputs are empty, clear UI elements
        actual_clear(False)
        actual_clear(True)
        return (
            gradio.update(value=None, visible=False),
            gradio.update(value=None, visible=False),
            gradio.update(value=None, visible=False),
            gradio.update(value=None, visible=False),
            gradio.update(visible=False),
            gradio.update(visible=False),
        )

    source_audio_1, source_image_1, audio_row_1, image_row_1 = actual_update([f.name for f in files1] if files1 else [])
    source_audio_2, source_image_2, audio_row_2, image_row_2 = actual_update([f.name for f in files2] if files2 else [],
                                                                             True)

    return source_audio_1, source_image_1, source_audio_2, source_image_2, audio_row_1, image_row_2
