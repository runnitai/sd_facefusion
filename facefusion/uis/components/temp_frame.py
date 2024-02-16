from typing import Optional, Tuple
import gradio

import facefusion.globals
import facefusion.choices
from facefusion import wording
from facefusion.typing import TempFrameFormat
from facefusion.filesystem import is_video
from facefusion.uis.core import get_ui_component, register_ui_component

TEMP_FRAME_FORMAT_DROPDOWN: Optional[gradio.Dropdown] = None
TEMP_FRAME_QUALITY_SLIDER: Optional[gradio.Slider] = None


def render() -> None:
    global TEMP_FRAME_FORMAT_DROPDOWN
    global TEMP_FRAME_QUALITY_SLIDER

    TEMP_FRAME_FORMAT_DROPDOWN = gradio.Dropdown(
        label = wording.get('uis.temp_frame_format_dropdown'),
        choices=facefusion.choices.temp_frame_formats,
        value=facefusion.globals.temp_frame_format,
        visible=is_video(facefusion.globals.target_path),
        elem_id='temp_frame_format_dropdown'
    )
    TEMP_FRAME_QUALITY_SLIDER = gradio.Slider(
        label = wording.get('uis.temp_frame_quality_slider'),
        value=facefusion.globals.temp_frame_quality,
        step=facefusion.choices.temp_frame_quality_range[1] - facefusion.choices.temp_frame_quality_range[0],
        minimum=facefusion.choices.temp_frame_quality_range[0],
        maximum=facefusion.choices.temp_frame_quality_range[-1],
        visible=is_video(facefusion.globals.target_path),
        elem_id='temp_frame_quality_slider'
    )
    register_ui_component('temp_frame_format_dropdown', TEMP_FRAME_FORMAT_DROPDOWN)
    register_ui_component('temp_frame_quality_slider', TEMP_FRAME_QUALITY_SLIDER)


def listen() -> None:
    TEMP_FRAME_FORMAT_DROPDOWN.select(update_temp_frame_format, inputs=TEMP_FRAME_FORMAT_DROPDOWN)
    TEMP_FRAME_QUALITY_SLIDER.change(update_temp_frame_quality, inputs=TEMP_FRAME_QUALITY_SLIDER)
    target_video = get_ui_component('target_video')
    if target_video:
        for method in ['upload', 'change', 'clear']:
            getattr(target_video, method)(remote_update,
                                          outputs=[TEMP_FRAME_FORMAT_DROPDOWN, TEMP_FRAME_QUALITY_SLIDER])


def remote_update() -> Tuple[gradio.update, gradio.update]:
    if is_video(facefusion.globals.target_path):
        return gradio.update(visible=True), gradio.update(visible=True)
    return gradio.update(visible=False), gradio.update(visible=False)


def update_temp_frame_format(temp_frame_format: TempFrameFormat) -> None:
    facefusion.globals.temp_frame_format = temp_frame_format


def update_temp_frame_quality(temp_frame_quality: int) -> None:
    facefusion.globals.temp_frame_quality = temp_frame_quality
