from typing import Optional, Tuple
import gradio

import facefusion.globals
import facefusion.choices
from facefusion import wording
from facefusion.typing import TempFrameFormat
from facefusion.filesystem import is_video
from facefusion.uis.core import get_ui_component, register_ui_component

TEMP_FRAME_FORMAT_DROPDOWN: Optional[gradio.Dropdown] = None


def render() -> None:
    global TEMP_FRAME_FORMAT_DROPDOWN

    TEMP_FRAME_FORMAT_DROPDOWN = gradio.Dropdown(
        label = wording.get('uis.temp_frame_format_dropdown'),
        choices=facefusion.choices.temp_frame_formats,
        value="bmp",
        visible=is_video(facefusion.globals.target_path),
        elem_id='temp_frame_format_dropdown'
    )


def listen() -> None:
    TEMP_FRAME_FORMAT_DROPDOWN.select(update_temp_frame_format, inputs=TEMP_FRAME_FORMAT_DROPDOWN)
    target_video = get_ui_component('target_video')
    if target_video:
        for method in ['upload', 'change', 'clear']:
            getattr(target_video, method)(remote_update, outputs=TEMP_FRAME_FORMAT_DROPDOWN)


def remote_update() -> gradio.Dropdown:
    if is_video(facefusion.globals.target_path):
        return gradio.update(visible=True)
    return gradio.update(visible=False)


def update_temp_frame_format(temp_frame_format: TempFrameFormat) -> None:
    facefusion.globals.temp_frame_format = temp_frame_format


