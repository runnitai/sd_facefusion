import os
from typing import Optional, List, Tuple, Union

import gradio

from facefusion import wording, state_manager
from facefusion.common_helper import get_first
from facefusion.filesystem import has_audio, has_image, filter_audio_paths, filter_image_paths, is_image
from facefusion.processors.modules.style_changer import process_src_image
from facefusion.uis.core import register_ui_component
from facefusion.uis.typing import File

SOURCE_FILE: Optional[gradio.File] = None
SOURCE_FILE_2: Optional[gradio.File] = None
SOURCE_AUDIO: Optional[gradio.Audio] = None
SOURCE_AUDIO_2: Optional[gradio.Audio] = None
SOURCE_IMAGE: Optional[gradio.Image] = None
SOURCE_IMAGE_2: Optional[gradio.Image] = None


def render() -> None:
    global SOURCE_FILE
    global SOURCE_FILE_2
    global SOURCE_AUDIO
    global SOURCE_AUDIO_2
    global SOURCE_IMAGE
    global SOURCE_IMAGE_2

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
    source_image_path = get_first(filter_image_paths(source_file_names))
    source_audio_path_2 = get_first(filter_audio_paths(source_file_names_2))
    source_image_path_2 = get_first(filter_image_paths(source_file_names_2))
    with gradio.Row():
        SOURCE_AUDIO = gradio.Audio(
            value=source_audio_path if has_source_audio else None,
            visible=has_source_audio,
            show_label=False
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


def check_swap_source_style(files: List[File], return_files: bool = False) -> Union[gradio.update, List[str]]:
    style_type = state_manager.get_item('style_changer_model')
    target = state_manager.get_item('style_changer_target')
    if target != 'source':
        file_names = [file.name for file in files if file] if files else None
        return gradio.update() if not return_files else file_names
    swapped_files = []
    if files:
        for file in files:
            if is_image(file.name):
                # Split the file name and extension, make sure it ends with _style_{style_type}
                file_name, file_extension = os.path.splitext(file.name)
                if "_style_" in file_name:
                    file_parts = file_name.split('_style_')
                    if len(file_parts) > 1 and file_parts[-1] == style_type:
                        swapped_files.append(file.name)
                    else:
                        # Find the original file
                        original_file = file_name.split('_style_')[0] + file_extension
                        if os.path.exists(original_file):
                            swapped = process_src_image(original_file, style_type)
                            swapped_files.append(swapped)
                        else:
                            print(f"Original file {original_file} not found.")
                else:
                    swapped = process_src_image(file.name, style_type)
                    swapped_files.append(swapped)
            else:
                swapped_files.append(file.name)
        return gradio.update(value=swapped_files) if not return_files else swapped_files
    return gradio.update() if not return_files else files


def update(files: List[File]) -> Tuple[gradio.Audio, gradio.update]:
    file_names = [file.name for file in files if file] if files else None
    if 'style_changer' in state_manager.get_item('processors'):
        files = check_swap_source_style(files, True)
        file_names = files

    has_source_audio = has_audio(file_names)
    has_source_image = has_image(file_names)
    # If we have a source_image, and style_changer is one of our frame_processors, we need to check the style_changer_target
    if has_source_audio or has_source_image:
        source_audio_path = get_first(filter_audio_paths(file_names))
        source_image_path = get_first(filter_image_paths(file_names))
        state_manager.set_item('source_paths', file_names)
        return gradio.update(value=source_audio_path, visible=has_source_audio), gradio.update(value=source_image_path,
                                                                                               visible=has_source_image)
    state_manager.clear_item('source_paths')
    return gradio.update(value=None, visible=False), gradio.update(value=None, visible=False)


def update_2(files: List[File]) -> Tuple[gradio.Audio, gradio.update]:
    file_names = [file.name for file in files] if files else None
    if 'style_changer' in state_manager.get_item('processors'):
        files = check_swap_source_style(files, True)
        file_names = files

    has_source_image = has_image(file_names)
    if has_source_image:
        source_image_path = get_first(filter_image_paths(file_names))
        state_manager.set_item('source_paths_2', file_names)
        return gradio.update(value=source_image_path, visible=has_source_image)
    state_manager.clear_item('source_paths_2')
    return gradio.update(value=None, visible=False)
