from time import sleep
from typing import Any, Dict, List, Optional

import cv2
import gradio

import facefusion.globals
from facefusion import wording, logger
from facefusion.audio import get_audio_frame
from facefusion.common_helper import get_first
from facefusion.content_analyser import analyse_frame
from facefusion.core import conditional_append_reference_faces
from facefusion.face_analyser import clear_face_analyser, get_average_face
from facefusion.face_store import clear_static_faces, get_reference_faces, clear_reference_faces
from facefusion.filesystem import is_video, is_image, filter_audio_paths
from facefusion.processors.frame.core import load_frame_processor_module
from facefusion.typing import Face, FaceSet, AudioFrame, VisionFrame
from facefusion.uis.components.face_masker import update_mask_buttons
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.uis.typing import ComponentName
from facefusion.vision import get_video_frame, count_video_frame_total, normalize_frame_color, \
    read_static_image, read_static_images, detect_fps, resize_frame_resolution

PREVIEW_IMAGE: Optional[gradio.Image] = None
PREVIEW_FRAME_SLIDER: Optional[gradio.Slider] = None
PREVIEW_FRAME_BACK_BUTTON: Optional[gradio.Button] = None
PREVIEW_FRAME_FORWARD_BUTTON: Optional[gradio.Button] = None
PREVIEW_FRAME_BACK_FIVE_BUTTON: Optional[gradio.Button] = None
PREVIEW_FRAME_FORWARD_FIVE_BUTTON: Optional[gradio.Button] = None


def render() -> None:
    global PREVIEW_IMAGE
    global PREVIEW_FRAME_SLIDER
    global PREVIEW_FRAME_BACK_BUTTON
    global PREVIEW_FRAME_FORWARD_BUTTON
    global PREVIEW_FRAME_BACK_FIVE_BUTTON
    global PREVIEW_FRAME_FORWARD_FIVE_BUTTON

    preview_image_args: Dict[str, Any] = \
        {
            'label': wording.get('uis.preview_image'),
            'interactive': False
        }
    preview_frame_slider_args: Dict[str, Any] = \
        {
            'label': wording.get('uis.preview_frame_slider'),
            'step': 1,
            'minimum': 0,
            'maximum': 100,
            'visible': False
        }
    conditional_append_reference_faces()
    reference_faces = get_reference_faces() if 'reference' in facefusion.globals.face_selector_mode else None
    source_frames = read_static_images(facefusion.globals.source_paths)
    source_face = get_average_face(source_frames)
    source_audio_path = get_first(filter_audio_paths(facefusion.globals.source_paths))
    if source_audio_path and facefusion.globals.output_video_fps:
        source_audio_frame = get_audio_frame(source_audio_path, facefusion.globals.output_video_fps,
                                             facefusion.globals.reference_frame_number)
    else:
        source_audio_frame = None
    if is_image(facefusion.globals.target_path):
        target_frame = read_static_image(facefusion.globals.target_path)
        preview_frame = process_preview_frame(reference_faces, source_face, source_audio_frame, target_frame, -1)
        preview_image_args['value'] = normalize_frame_color(preview_frame)
    if is_video(facefusion.globals.target_path):
        frame_number = facefusion.globals.reference_frame_number
        temp_frame = get_video_frame(facefusion.globals.target_path, frame_number)
        preview_frame = process_preview_frame(reference_faces, source_face, source_audio_frame, temp_frame, frame_number)
        preview_image_args['value'] = normalize_frame_color(preview_frame)
        preview_image_args['visible'] = True
        preview_frame_slider_args['value'] = facefusion.globals.reference_frame_number
        preview_frame_slider_args['maximum'] = count_video_frame_total(facefusion.globals.target_path)
        preview_frame_slider_args['visible'] = True
    PREVIEW_IMAGE = gradio.Image(**preview_image_args)
    with gradio.Row():
        PREVIEW_FRAME_BACK_FIVE_BUTTON = gradio.Button(
            value="-5s",
            elem_id='ff_preview_frame_back_five_button',
            elem_classes=['ff_preview_frame_button'],
            visible=preview_frame_slider_args['visible']
        )

        PREVIEW_FRAME_BACK_BUTTON = gradio.Button(
            value="-1s",
            elem_id='ff_preview_frame_back_button',
            elem_classes=['ff_preview_frame_button'],
            visible=preview_frame_slider_args['visible']
        )
        PREVIEW_FRAME_SLIDER = gradio.Slider(**preview_frame_slider_args)

        PREVIEW_FRAME_FORWARD_BUTTON = gradio.Button(
            value="+1s",
            elem_id='ff_preview_frame_forward_button',
            elem_classes=['ff_preview_frame_button'],
            visible=preview_frame_slider_args['visible']
        )
        PREVIEW_FRAME_FORWARD_FIVE_BUTTON = gradio.Button(
            value="+5s",
            elem_id='ff_preview_frame_forward_five_button',
            elem_classes=['ff_preview_frame_button'],
            visible=preview_frame_slider_args['visible']
        )

    register_ui_component('preview_frame_slider', PREVIEW_FRAME_SLIDER)
    register_ui_component('preview_frame_back_button', PREVIEW_FRAME_BACK_BUTTON)
    register_ui_component('preview_frame_forward_button', PREVIEW_FRAME_FORWARD_BUTTON)
    register_ui_component('preview_frame_back_five_button', PREVIEW_FRAME_BACK_FIVE_BUTTON)
    register_ui_component('preview_frame_forward_five_button', PREVIEW_FRAME_FORWARD_FIVE_BUTTON)
    register_ui_component('preview_image', PREVIEW_IMAGE)


def listen() -> None:
    mask_disable_button = get_ui_component('mask_disable_button')
    mask_enable_button = get_ui_component('mask_enable_button')
    mask_clear = get_ui_component('mask_clear_button')
    all_update_elements = [PREVIEW_IMAGE, mask_enable_button, mask_disable_button]
    more_elements = [PREVIEW_FRAME_SLIDER] + all_update_elements
    PREVIEW_FRAME_BACK_BUTTON.click(preview_back, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements)
    PREVIEW_FRAME_BACK_FIVE_BUTTON.click(preview_back_five, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements)
    PREVIEW_FRAME_FORWARD_BUTTON.click(preview_forward, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements)
    PREVIEW_FRAME_FORWARD_FIVE_BUTTON.click(preview_forward_five, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements)
    PREVIEW_FRAME_SLIDER.input(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements)
    mask_disable_button.click(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements)
    mask_enable_button.click(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements)
    mask_clear.click(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements)
    multi_one_component_names: List[ComponentName] = \
        [
            'source_audio',
            'source_image',
            'target_image',
            'target_video'
        ]
    for component_name in multi_one_component_names:
        component = get_ui_component(component_name)
        if component:
            for method in ['upload', 'change', 'clear']:
                getattr(component, method)(update_preview_image, inputs=PREVIEW_FRAME_SLIDER,
                                           outputs=all_update_elements)
    multi_two_component_names: List[ComponentName] = \
        [
            'target_image',
            'target_video'
        ]
    for component_name in multi_two_component_names:
        component = get_ui_component(component_name)
        if component:
            for method in ['upload', 'change', 'clear']:
                getattr(component, method)(update_preview_frame_slider,
                                           outputs=[PREVIEW_FRAME_SLIDER, PREVIEW_FRAME_BACK_BUTTON,
                                                    PREVIEW_FRAME_FORWARD_BUTTON, PREVIEW_FRAME_BACK_FIVE_BUTTON,
                                                    PREVIEW_FRAME_FORWARD_FIVE_BUTTON])
    select_component_names: List[ComponentName] = \
        [
            'face_analyser_order_dropdown',
            'face_analyser_age_dropdown',
            'face_analyser_gender_dropdown'
        ]
    for component_name in select_component_names:
        component = get_ui_component(component_name)
        if component:
            component.select(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements)
    change_one_component_names: List[ComponentName] = \
        [
            'face_debugger_items_checkbox_group',
            'face_enhancer_blend_slider',
            'frame_enhancer_blend_slider',
            'face_selector_mode_dropdown',
            'reference_face_distance_slider',
            'face_mask_types_checkbox_group',
            'face_mask_blur_slider',
            'face_mask_padding_top_slider',
            'face_mask_padding_bottom_slider',
            'face_mask_padding_left_slider',
            'face_mask_padding_right_slider',
            'face_mask_region_checkbox_group',
            'face_analyser_order_dropdown',
            'face_analyser_age_dropdown',
            'face_analyser_gender_dropdown',
            'output_video_fps_slider'
        ]
    for component_name in change_one_component_names:
        component = get_ui_component(component_name)
        if component:
            component.change(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements)
    change_two_component_names: List[ComponentName] = \
    [
        'frame_processors_checkbox_group',
        'face_enhancer_model_dropdown',
        'face_swapper_model_dropdown',
        'frame_enhancer_model_dropdown',
        'lip_syncer_model_dropdown',
        'face_detector_model_dropdown',
        'face_detector_size_dropdown',
        'face_detector_score_slider'
    ]
    for component_name in change_two_component_names:
        component = get_ui_component(component_name)
        if component:
            component.change(clear_and_update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements)


def clear_and_update_preview_image(frame_number: int = 0) -> gradio.Image:
    clear_face_analyser()
    clear_reference_faces()
    clear_static_faces()
    sleep(0.5)
    return update_preview_image(frame_number)


def update_preview_image(frame_number: int = 0) -> gradio.Image:
    global_processors = facefusion.globals.frame_processors
    from facefusion.uis.components.frame_processors import sort_frame_processors
    global_processors = sort_frame_processors(global_processors)
    for frame_processor in global_processors:
        frame_processor_module = load_frame_processor_module(frame_processor)
        while not frame_processor_module.post_check():
            logger.disable()
            sleep(0.5)
        logger.enable()
    conditional_append_reference_faces()
    source_frames = read_static_images(facefusion.globals.source_paths)
    source_face = get_average_face(source_frames)
    source_audio_path = get_first(filter_audio_paths(facefusion.globals.source_paths))
    if source_audio_path and facefusion.globals.output_video_fps:
        source_audio_frame = get_audio_frame(source_audio_path, facefusion.globals.output_video_fps, frame_number)
    else:
        source_audio_frame = None

    enable_button, disable_button = update_mask_buttons(frame_number)
    reference_faces = get_reference_faces() if 'reference' in facefusion.globals.face_selector_mode else None
    if is_image(facefusion.globals.target_path):
        target_frame = read_static_image(facefusion.globals.target_path)
        preview_frame = process_preview_frame(reference_faces, source_face, source_audio_frame, target_frame, -1)
        preview_frame = normalize_frame_color(preview_frame)
        return gradio.update(value=preview_frame, visible=True), enable_button, disable_button
    if is_video(facefusion.globals.target_path):
        temp_frame = get_video_frame(facefusion.globals.target_path, frame_number)
        preview_frame = process_preview_frame(reference_faces, source_face, source_audio_frame, temp_frame, frame_number)
        preview_frame = normalize_frame_color(preview_frame)
        return gradio.update(value=preview_frame, visible=True), enable_button, disable_button
    return gradio.update(value=None, visible=True), enable_button, disable_button


def preview_back(reference_frame_number: int = 0) -> gradio.update:
    frames_per_second = int(detect_fps(facefusion.globals.target_path))
    reference_frame_number = max(0, reference_frame_number - frames_per_second)
    preview, enable_btn, disable_btn = update_preview_image(reference_frame_number)
    return gradio.update(value=reference_frame_number), preview, enable_btn, disable_btn


def preview_forward(reference_frame_number: int = 0) -> gradio.update:
    frames_per_second = int(detect_fps(facefusion.globals.target_path))
    reference_frame_number = min(reference_frame_number + frames_per_second,
                                 count_video_frame_total(facefusion.globals.target_path))
    preview, enable_btn, disable_btn = update_preview_image(reference_frame_number)
    return gradio.update(value=reference_frame_number), preview, enable_btn, disable_btn


def preview_back_five(reference_frame_number: int = 0) -> gradio.update:
    frames_per_second = int(detect_fps(facefusion.globals.target_path)) * 5
    reference_frame_number = max(0, reference_frame_number - frames_per_second)
    preview, enable_btn, disable_btn = update_preview_image(reference_frame_number)
    return gradio.update(value=reference_frame_number), preview, enable_btn, disable_btn


def preview_forward_five(reference_frame_number: int = 0) -> gradio.update:
    frames_per_second = int(detect_fps(facefusion.globals.target_path)) * 5
    reference_frame_number = min(reference_frame_number + frames_per_second,
                                 count_video_frame_total(facefusion.globals.target_path))
    preview, enable_btn, disable_btn = update_preview_image(reference_frame_number)
    return gradio.update(value=reference_frame_number), preview, enable_btn, disable_btn


def update_preview_frame_slider() -> gradio.update:
    if is_video(facefusion.globals.target_path):
        video_frame_total = count_video_frame_total(facefusion.globals.target_path)
        return gradio.update(maximum=video_frame_total, visible=True), gradio.update(visible=True), gradio.update(
            visible=True), gradio.update(visible=True), gradio.update(visible=True)
    return gradio.update(value=None, maximum=None, visible=False), gradio.update(visible=False), gradio.update(
        visible=False), gradio.update(visible=False), gradio.update(visible=False)


def process_preview_frame(reference_faces: FaceSet, source_face: Face, source_audio_frame: AudioFrame,
                          target_vision_frame: VisionFrame, frame_number=-1) -> VisionFrame:
    target_vision_frame = resize_frame_resolution(target_vision_frame, 640, 640)
    if analyse_frame(target_vision_frame):
        return cv2.GaussianBlur(target_vision_frame, (99, 99), 0)
    global_processors = facefusion.globals.frame_processors
    # Sort global_processors so 'face_debugger' is last if it's in global_processors
    global_processors = sorted(
        global_processors,
        key=lambda fp: (
            global_processors.index(fp) if fp in global_processors else len(global_processors),
            fp == "face_debugger"
        )
    )
    source_frame = target_vision_frame.copy()
    for frame_processor in global_processors:
        print("Processing with frame processor: ", frame_processor)
        frame_processor_module = load_frame_processor_module(frame_processor)
        logger.disable()
        if frame_processor_module.pre_process('preview'):
            logger.enable()
            target_vision_frame = frame_processor_module.process_frame(
                {
                    'reference_faces': reference_faces,
                    'source_face': source_face,
                    'source_audio_frame': source_audio_frame,
                    'target_vision_frame': target_vision_frame,
                    'target_frame_number': frame_number,
                    'source_frame': source_frame,
                })
        # Apply overlay to temp_frame

    return target_vision_frame
