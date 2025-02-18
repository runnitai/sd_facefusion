from typing import Optional, Tuple

import gradio

import facefusion.globals
from facefusion import wording, choices, state_manager
from facefusion.common_helper import calc_int_step
from facefusion.filesystem import is_image, is_video, get_output_path_auto
from facefusion.typing import OutputVideoEncoder, OutputVideoPreset, Fps, OutputAudioEncoder
from facefusion.uis.core import register_ui_component, get_ui_components
from facefusion.vision import create_video_resolutions, detect_video_resolution, pack_resolution, detect_video_fps, \
    detect_image_resolution, create_image_resolutions

OUTPUT_PATH_TEXTBOX: Optional[gradio.Textbox] = None
OUTPUT_IMAGE_QUALITY_SLIDER: Optional[gradio.Slider] = None
OUTPUT_IMAGE_RESOLUTION_DROPDOWN: Optional[gradio.Dropdown] = None
OUTPUT_AUDIO_ENCODER_DROPDOWN: Optional[gradio.Dropdown] = None
OUTPUT_VIDEO_ENCODER_DROPDOWN: Optional[gradio.Dropdown] = None
OUTPUT_VIDEO_PRESET_DROPDOWN: Optional[gradio.Dropdown] = None
OUTPUT_VIDEO_RESOLUTION_DROPDOWN: Optional[gradio.Dropdown] = None
OUTPUT_VIDEO_QUALITY_SLIDER: Optional[gradio.Slider] = None
OUTPUT_VIDEO_FPS_SLIDER: Optional[gradio.Slider] = None


def render() -> None:
    global OUTPUT_PATH_TEXTBOX
    global OUTPUT_IMAGE_QUALITY_SLIDER
    global OUTPUT_IMAGE_RESOLUTION_DROPDOWN
    global OUTPUT_AUDIO_ENCODER_DROPDOWN
    global OUTPUT_VIDEO_ENCODER_DROPDOWN
    global OUTPUT_VIDEO_PRESET_DROPDOWN
    global OUTPUT_VIDEO_RESOLUTION_DROPDOWN
    global OUTPUT_VIDEO_QUALITY_SLIDER
    global OUTPUT_VIDEO_FPS_SLIDER

    output_image_resolutions = []
    output_video_resolutions = []
    if is_image(state_manager.get_item('target_path')):
        output_image_resolution = detect_image_resolution(state_manager.get_item('target_path'))
        output_image_resolutions = create_image_resolutions(output_image_resolution)
    if is_video(state_manager.get_item('target_path')):
        output_video_resolution = detect_video_resolution(state_manager.get_item('target_path'))
        output_video_resolutions = create_video_resolutions(output_video_resolution)
    out_path = get_output_path_auto()
    #out_path = os.path.join(script_path, out_dir, 'facefusion')
    state_manager.init_item('output_path', out_path)

    OUTPUT_PATH_TEXTBOX = gradio.Textbox(
        label=wording.get('uis.output_path_textbox'),
        value=out_path,
        max_lines=1,
        elem_id='output_path_textbox',
        visible=False
    )
    OUTPUT_IMAGE_QUALITY_SLIDER = gradio.Slider(
        label=wording.get('uis.output_image_quality_slider'),
        value=state_manager.get_item('output_image_quality'),
        step=calc_int_step(facefusion.choices.output_image_quality_range),
        minimum=facefusion.choices.output_image_quality_range[0],
        maximum=facefusion.choices.output_image_quality_range[-1],
        visible=is_image(state_manager.get_item('target_path'))
    )
    OUTPUT_IMAGE_RESOLUTION_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.output_image_resolution_dropdown'),
        choices=output_image_resolutions,
        value=state_manager.get_item('output_image_resolution'),
        visible=is_image(state_manager.get_item('target_path'))
    )
    OUTPUT_AUDIO_ENCODER_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.output_audio_encoder_dropdown'),
        choices=facefusion.choices.output_audio_encoders,
        value=state_manager.get_item('output_audio_encoder'),
        visible=is_video(state_manager.get_item('target_path'))
    )
    OUTPUT_VIDEO_ENCODER_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.output_video_encoder_dropdown'),
        choices=facefusion.choices.output_video_encoders,
        value=state_manager.get_item('output_video_encoder'),
        visible=is_video(state_manager.get_item('target_path'))
    )
    OUTPUT_VIDEO_PRESET_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.output_video_preset_dropdown'),
        choices=facefusion.choices.output_video_presets,
        value=state_manager.get_item('output_video_preset'),
        visible=is_video(state_manager.get_item('target_path'))
    )
    OUTPUT_VIDEO_QUALITY_SLIDER = gradio.Slider(
        label=wording.get('uis.output_video_quality_slider'),
        value=state_manager.get_item('output_video_quality'),
        step=calc_int_step(facefusion.choices.output_video_quality_range),
        minimum=facefusion.choices.output_video_quality_range[0],
        maximum=facefusion.choices.output_video_quality_range[-1],
        visible=is_video(state_manager.get_item('target_path'))
    )
    OUTPUT_VIDEO_RESOLUTION_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.output_video_resolution_dropdown'),
        choices=output_video_resolutions,
        value=state_manager.get_item('output_video_resolution'),
        visible=is_video(state_manager.get_item('target_path'))
    )
    OUTPUT_VIDEO_FPS_SLIDER = gradio.Slider(
        label=wording.get('uis.output_video_fps_slider'),
        value=state_manager.get_item('output_video_fps'),
        step=0.01,
        minimum=1,
        maximum=60,
        visible=is_video(state_manager.get_item('target_path'))
    )

    register_ui_component('output_path_textbox', OUTPUT_PATH_TEXTBOX)
    register_ui_component('output_image_quality_slider', OUTPUT_IMAGE_QUALITY_SLIDER)
    register_ui_component('output_video_encoder_dropdown', OUTPUT_VIDEO_ENCODER_DROPDOWN)
    register_ui_component('output_video_preset_dropdown', OUTPUT_VIDEO_PRESET_DROPDOWN)
    register_ui_component('output_video_quality_slider', OUTPUT_VIDEO_QUALITY_SLIDER)
    register_ui_component('output_video_resolution_dropdown', OUTPUT_VIDEO_RESOLUTION_DROPDOWN)
    register_ui_component('output_video_fps_slider', OUTPUT_VIDEO_FPS_SLIDER)
    register_ui_component('output_image_resolution_dropdown', OUTPUT_IMAGE_RESOLUTION_DROPDOWN)
    register_ui_component('output_audio_encoder_dropdown', OUTPUT_AUDIO_ENCODER_DROPDOWN)



def listen() -> None:
    OUTPUT_PATH_TEXTBOX.change(update_output_path, inputs=OUTPUT_PATH_TEXTBOX)
    OUTPUT_IMAGE_QUALITY_SLIDER.release(update_output_image_quality, inputs=OUTPUT_IMAGE_QUALITY_SLIDER)
    OUTPUT_IMAGE_RESOLUTION_DROPDOWN.change(update_output_image_resolution, inputs=OUTPUT_IMAGE_RESOLUTION_DROPDOWN)
    OUTPUT_AUDIO_ENCODER_DROPDOWN.change(update_output_audio_encoder, inputs=OUTPUT_AUDIO_ENCODER_DROPDOWN)
    OUTPUT_VIDEO_ENCODER_DROPDOWN.change(update_output_video_encoder, inputs=OUTPUT_VIDEO_ENCODER_DROPDOWN)
    OUTPUT_VIDEO_PRESET_DROPDOWN.change(update_output_video_preset, inputs=OUTPUT_VIDEO_PRESET_DROPDOWN)
    OUTPUT_VIDEO_QUALITY_SLIDER.release(update_output_video_quality, inputs=OUTPUT_VIDEO_QUALITY_SLIDER)
    OUTPUT_VIDEO_RESOLUTION_DROPDOWN.change(update_output_video_resolution, inputs=OUTPUT_VIDEO_RESOLUTION_DROPDOWN)
    OUTPUT_VIDEO_FPS_SLIDER.release(update_output_video_fps, inputs=OUTPUT_VIDEO_FPS_SLIDER)
    for ui_component in get_ui_components(
        [
            'target_image',
            'target_video'
        ]):
        for method in ['upload', 'change', 'clear']:
            getattr(ui_component, method)(remote_update,
                                          outputs=[OUTPUT_IMAGE_QUALITY_SLIDER, OUTPUT_IMAGE_RESOLUTION_DROPDOWN,
                                                   OUTPUT_AUDIO_ENCODER_DROPDOWN, OUTPUT_VIDEO_ENCODER_DROPDOWN,
                                                   OUTPUT_VIDEO_PRESET_DROPDOWN, OUTPUT_VIDEO_QUALITY_SLIDER,
                                                   OUTPUT_VIDEO_RESOLUTION_DROPDOWN, OUTPUT_VIDEO_FPS_SLIDER])


def remote_update() -> Tuple[
    gradio.update, gradio.update, gradio.update, gradio.update, gradio.update, gradio.update, gradio.update, gradio.update]:
    if is_image(state_manager.get_item('target_path')):
        output_image_resolution = detect_image_resolution(state_manager.get_item('target_path'))
        output_image_resolutions = create_image_resolutions(output_image_resolution)
        state_manager.set_item('output_image_resolution', pack_resolution(output_image_resolution))
        return gradio.update(visible=True), gradio.update(value=state_manager.get_item('output_image_resolution'),
                                                            choices=output_image_resolutions,
                                                            visible=True), gradio.update(
            visible=False), gradio.update(visible=False), gradio.update(visible=False), gradio.update(
            visible=False), gradio.update(visible=False), gradio.update(visible=False)
    if is_video(state_manager.get_item('target_path')):
        output_video_resolution = detect_video_resolution(state_manager.get_item('target_path'))
        output_video_resolutions = create_video_resolutions(output_video_resolution)
        state_manager.set_item('output_video_resolution', pack_resolution(output_video_resolution))
        state_manager.set_item('output_video_fps', detect_video_fps(state_manager.get_item('target_path')))
        return gradio.update(visible=False), gradio.update(visible=False), gradio.update(
            visible=True), gradio.update(visible=True), gradio.update(visible=True), gradio.update(
            visible=True), gradio.update(value=state_manager.get_item('output_video_resolution'),
                                           choices=output_video_resolutions, visible=True), gradio.update(
            value=state_manager.get_item('output_video_fps'), visible=True)
    return gradio.update(visible=False), gradio.update(visible=False), gradio.update(
        visible=False), gradio.update(visible=False), gradio.update(visible=False), gradio.update(
        visible=False), gradio.update(visible=False), gradio.update(visible=False)


def update_output_path(output_path: str) -> None:
    state_manager.set_item('output_path', output_path)


def update_output_image_quality(output_image_quality: float) -> None:
    state_manager.set_item('output_image_quality', int(output_image_quality))


def update_output_image_resolution(output_image_resolution: str) -> None:
    state_manager.set_item('output_image_resolution', output_image_resolution)


def update_output_audio_encoder(output_audio_encoder: OutputAudioEncoder) -> None:
    state_manager.set_item('output_audio_encoder', output_audio_encoder)


def update_output_video_encoder(output_video_encoder: OutputVideoEncoder) -> None:
    state_manager.set_item('output_video_encoder', output_video_encoder)


def update_output_video_preset(output_video_preset: OutputVideoPreset) -> None:
    state_manager.set_item('output_video_preset', output_video_preset)


def update_output_video_quality(output_video_quality: float) -> None:
    state_manager.set_item('output_video_quality', int(output_video_quality))


def update_output_video_resolution(output_video_resolution: str) -> None:
    state_manager.set_item('output_video_resolution', output_video_resolution)


def update_output_video_fps(output_video_fps: Fps) -> None:
    state_manager.set_item('output_video_fps', output_video_fps)
