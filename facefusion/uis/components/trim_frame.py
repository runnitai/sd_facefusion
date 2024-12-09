from typing import Optional, Tuple

import gradio
from gradio import Slider

from facefusion import state_manager, wording
from facefusion.face_store import clear_static_faces
from facefusion.filesystem import is_video
from facefusion.uis.core import get_ui_components, register_ui_component

from facefusion.uis.typing import ComponentOptions
from facefusion.vision import count_video_frame_total

TRIM_FRAME_START_SLIDER: Optional[Slider] = None
TRIM_FRAME_END_SLIDER: Optional[Slider] = None


# TRIM_FRAME_RANGE_SLIDER: Optional[RangeSlider] = None


def render() -> None:
    global TRIM_FRAME_START_SLIDER
    global TRIM_FRAME_END_SLIDER
    # global TRIM_FRAME_RANGE_SLIDER

    trim_frame_range_slider_options: ComponentOptions = \
        {
            'label': wording.get('uis.trim_frame_slider'),
            'minimum': 0,
            'step': 1,
            'visible': False
        }
    if is_video(state_manager.get_item('target_path')):
        video_frame_total = count_video_frame_total(state_manager.get_item('target_path'))
        trim_frame_start = state_manager.get_item('trim_frame_start') or 0
        trim_frame_end = state_manager.get_item('trim_frame_end') or video_frame_total
        trim_frame_range_slider_options['maximum'] = video_frame_total
        trim_frame_range_slider_options['value'] = (trim_frame_start, trim_frame_end)
        trim_frame_range_slider_options['visible'] = True
    with gradio.Group():
        with gradio.Row():
            TRIM_FRAME_START_SLIDER = Slider(
                label=wording.get('uis.trim_frame_start_slider'),
                minimum=0,
                step=1,
                visible=False
            )

            TRIM_FRAME_END_SLIDER = Slider(
                label=wording.get('uis.trim_frame_end_slider'),
                minimum=0,
                step=1,
                visible=False
            )
    register_ui_component('trim_frame_start_slider', TRIM_FRAME_START_SLIDER)
    register_ui_component('trim_frame_end_slider', TRIM_FRAME_END_SLIDER)


def listen() -> None:
    TRIM_FRAME_START_SLIDER.release(update_trim_frame_start, inputs=TRIM_FRAME_START_SLIDER)
    TRIM_FRAME_END_SLIDER.release(update_trim_frame_end, inputs=TRIM_FRAME_END_SLIDER)
    for ui_component in get_ui_components(
            [
                'target_image',
                'target_image_2'
                'target_video'
            ]):
        for method in ['upload', 'change', 'clear']:
            getattr(ui_component, method)(remote_update, outputs=[TRIM_FRAME_START_SLIDER, TRIM_FRAME_END_SLIDER])


def remote_update() -> Tuple[gradio.update, gradio.update]:
    print('remote_update (trimframe)')
    if is_video(state_manager.get_item('target_path')):
        print('remote_update (trimframe) - is_video')
        video_frame_total = count_video_frame_total(state_manager.get_item('target_path'))
        state_manager.clear_item('trim_frame_start')
        state_manager.clear_item('trim_frame_end')
        return gradio.update(value=0, maximum=video_frame_total, visible=True), gradio.update(value=video_frame_total,
                                                                                maximum=video_frame_total, visible=True)
    return gradio.update(value=0, maximum=0, visible=False), gradio.update(value=0, maximum=0, visible=False)


def update_trim_frame(trim_frame: Tuple[float, float]) -> None:
    clear_static_faces()
    trim_frame_start, trim_frame_end = trim_frame
    video_frame_total = count_video_frame_total(state_manager.get_item('target_path'))
    trim_frame_start = int(trim_frame_start) if trim_frame_start > 0 else None
    trim_frame_end = int(trim_frame_end) if trim_frame_end < video_frame_total else None
    state_manager.set_item('trim_frame_start', trim_frame_start)
    state_manager.set_item('trim_frame_end', trim_frame_end)


def update_trim_frame_start(trim_frame_start: float) -> None:
    clear_static_faces()
    video_frame_total = count_video_frame_total(state_manager.get_item('target_path'))
    trim_frame_start = int(trim_frame_start) if trim_frame_start > 0 else None
    state_manager.set_item('trim_frame_start', trim_frame_start)


def update_trim_frame_end(trim_frame_end: float) -> None:
    clear_static_faces()
    video_frame_total = count_video_frame_total(state_manager.get_item('target_path'))
    trim_frame_end = int(trim_frame_end) if trim_frame_end < video_frame_total else None
    state_manager.set_item('trim_frame_end', trim_frame_end)
