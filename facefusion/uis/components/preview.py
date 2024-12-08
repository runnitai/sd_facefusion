import threading
import traceback
from time import sleep
from typing import Any, Dict, Optional, Tuple

import cv2
import gradio

from facefusion import wording, process_manager, state_manager
from facefusion.audio import get_audio_frame, create_empty_audio_frame
from facefusion.common_helper import get_first
from facefusion.content_analyser import analyse_frame
from facefusion.core import conditional_append_reference_faces
from facefusion.face_analyser import get_avg_faces
from facefusion.face_store import clear_static_faces, get_reference_faces, clear_reference_faces
from facefusion.filesystem import is_video, is_image, filter_audio_paths
from facefusion.processors.core import load_processor_module
from facefusion.typing import Face, FaceSet, AudioFrame, VisionFrame
from facefusion.uis.components.face_masker import update_mask_buttons
from facefusion.uis.core import get_ui_component, register_ui_component, get_ui_components
from facefusion.vision import get_video_frame, count_video_frame_total, normalize_frame_color, \
    read_static_image, detect_video_fps, resize_frame_resolution

PREVIEW_IMAGE: Optional[gradio.Image] = None
PREVIEW_FRAME_SLIDER: Optional[gradio.Slider] = None
PREVIEW_FRAME_BACK_BUTTON: Optional[gradio.Button] = None
PREVIEW_FRAME_FORWARD_BUTTON: Optional[gradio.Button] = None
PREVIEW_FRAME_BACK_FIVE_BUTTON: Optional[gradio.Button] = None
PREVIEW_FRAME_FORWARD_FIVE_BUTTON: Optional[gradio.Button] = None
PREVIEW_FRAME_ROW: Optional[gradio.Row] = None

CURRENT_PREVIEW_FRAME_NUMBER = -1
AVG_FACE_1 = None
AVG_FACE_2 = None
SOURCE_FRAMES_1 = []
SOURCE_FRAMES_2 = []

frame_processing_lock = threading.Lock()


def render() -> None:
    global PREVIEW_IMAGE
    global PREVIEW_FRAME_SLIDER
    global PREVIEW_FRAME_BACK_BUTTON
    global PREVIEW_FRAME_FORWARD_BUTTON
    global PREVIEW_FRAME_BACK_FIVE_BUTTON
    global PREVIEW_FRAME_FORWARD_FIVE_BUTTON
    global PREVIEW_FRAME_ROW

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
    reference_faces, reference_faces_2 = (
        get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else (None, None))

    source_face, source_face_2 = get_avg_faces()
    source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
    source_audio_path_2 = get_first(filter_audio_paths(state_manager.get_item('source_paths_2')))
    if source_audio_path and state_manager.get_item('output_video_fps'):
        source_audio_frame = get_audio_frame(source_audio_path, state_manager.get_item('output_video_fps'), 0)
    else:
        source_audio_frame = None
    if source_audio_path_2 and state_manager.get_item('output_video_fps'):
        source_audio_frame_2 = get_audio_frame(source_audio_path_2, state_manager.get_item('output_video_fps'), 0)
    else:
        source_audio_frame_2 = None
    target_path = state_manager.get_item('target_path')
    if is_image(target_path):
        target_frame = read_static_image(target_path)
        preview_frame = process_preview_frame(reference_faces, reference_faces_2, source_face, source_face_2,
                                              source_audio_frame, source_audio_frame_2, target_frame, -1)
        preview_image_args['value'] = normalize_frame_color(preview_frame)

    if is_video(target_path):
        frame_number = 0
        temp_frame = get_video_frame(target_path, frame_number)
        preview_frame = process_preview_frame(reference_faces, reference_faces_2, source_face, source_face_2,
                                              source_audio_frame, source_audio_frame_2, temp_frame, frame_number)
        preview_image_args['value'] = normalize_frame_color(preview_frame)
        preview_image_args['visible'] = True
        preview_frame_slider_args['value'] = 0
        preview_frame_slider_args['maximum'] = count_video_frame_total(target_path)
        preview_frame_slider_args['visible'] = True

    PREVIEW_IMAGE = gradio.Image(**preview_image_args)
    with gradio.Row() as PREVIEW_FRAME_ROW:
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
    register_ui_component('preview_frame_row', PREVIEW_FRAME_ROW)


def listen() -> None:
    mask_disable_button = get_ui_component('mask_disable_button')
    mask_enable_button = get_ui_component('mask_enable_button')
    mask_clear = get_ui_component('mask_clear_button')
    all_update_elements = [PREVIEW_IMAGE, mask_enable_button, mask_disable_button]
    more_elements = [PREVIEW_FRAME_SLIDER] + all_update_elements
    PREVIEW_FRAME_BACK_BUTTON.click(preview_back, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements, show_progress='hidden')
    PREVIEW_FRAME_BACK_FIVE_BUTTON.click(preview_back_five, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements,
                                         show_progress='hidden')
    PREVIEW_FRAME_FORWARD_BUTTON.click(preview_forward, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements, show_progress='hidden')
    PREVIEW_FRAME_FORWARD_FIVE_BUTTON.click(preview_forward_five, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements,
                                            show_progress='hidden')
    PREVIEW_FRAME_SLIDER.release(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements,
                                 show_progress='hidden')
    # PREVIEW_FRAME_SLIDER.change(_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=[PREVIEW_IMAGE],
    #                             show_progress='hidden')
    mask_disable_button.click(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements,
                              show_progress='hidden')
    mask_enable_button.click(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements,
                             show_progress='hidden')
    mask_clear.click(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements, show_progress='hidden')
    for ui_component in get_ui_components(
            [
                'source_audio',
                'source_image',
                'source_image_2',
                'target_image',
                'target_video'
            ]):
        for method in ['upload', 'change', 'clear']:
            getattr(ui_component, method)(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements,
                                          show_progress='hidden')

    for ui_component in get_ui_components(
            [
                'target_image',
                'target_video'
            ]):
        for method in ['upload', 'change', 'clear']:
            getattr(ui_component, method)(update_preview_frame_slider,
                                          outputs=[PREVIEW_FRAME_SLIDER, PREVIEW_FRAME_BACK_BUTTON,
                                                   PREVIEW_FRAME_FORWARD_BUTTON, PREVIEW_FRAME_BACK_FIVE_BUTTON,
                                                   PREVIEW_FRAME_FORWARD_FIVE_BUTTON, PREVIEW_FRAME_ROW])
    for ui_component in get_ui_components(
            [
                'face_debugger_items_checkbox_group',
                'frame_colorizer_size_dropdown',
                'face_mask_types_checkbox_group',
                'face_mask_regions_checkbox_group'
            ]):
        ui_component.change(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements,
                            show_progress='hidden')
    for ui_component in get_ui_components(
            [
                'age_modifier_direction_slider',
                'expression_restorer_factor_slider',
                'face_editor_eyebrow_direction_slider',
                'face_editor_eye_gaze_horizontal_slider',
                'face_editor_eye_gaze_vertical_slider',
                'face_editor_eye_open_ratio_slider',
                'face_editor_lip_open_ratio_slider',
                'face_editor_mouth_grim_slider',
                'face_editor_mouth_pout_slider',
                'face_editor_mouth_purse_slider',
                'face_editor_mouth_smile_slider',
                'face_editor_mouth_position_horizontal_slider',
                'face_editor_mouth_position_vertical_slider',
                'face_editor_head_pitch_slider',
                'face_editor_head_yaw_slider',
                'face_editor_head_roll_slider',
                'face_enhancer_blend_slider',
                'frame_colorizer_blend_slider',
                'frame_enhancer_blend_slider',
                'reference_face_distance_slider',
                'face_selector_age_range_slider',
                'face_mask_blur_slider',
                'face_mask_padding_top_slider',
                'face_mask_padding_bottom_slider',
                'face_mask_padding_left_slider',
                'face_mask_padding_right_slider',
                'output_video_fps_slider'
            ]):
        if ui_component:
            ui_component.release(update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=all_update_elements)
    for ui_component in get_ui_components(
            [
                'age_modifier_model_dropdown',
                'expression_restorer_model_dropdown',
                'processors_checkbox_group',
                'face_editor_model_dropdown',
                'face_enhancer_model_dropdown',
                'face_swapper_model_dropdown',
                'face_swapper_pixel_boost_dropdown',
                'frame_colorizer_model_dropdown',
                'frame_enhancer_model_dropdown',
                'lip_syncer_model_dropdown',
                'face_selector_mode_dropdown',
                'face_selector_order_dropdown',
                'face_selector_gender_dropdown',
                'face_selector_race_dropdown',
                'face_detector_model_dropdown',
                'face_detector_size_dropdown',
                'face_detector_angles_checkbox_group',
                'face_landmarker_model_dropdown'
            ]):
        if ui_component:
            ui_component.change(clear_and_update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=PREVIEW_IMAGE)

    for ui_component in get_ui_components(
            [
                'face_detector_score_slider',
                'face_landmarker_score_slider'
            ]):
        ui_component.release(clear_and_update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=PREVIEW_IMAGE)


def clear_and_update_preview_image(frame_number: int = 0) -> gradio.update:
    global CURRENT_PREVIEW_FRAME_NUMBER
    CURRENT_PREVIEW_FRAME_NUMBER = -1
    clear_reference_faces()
    clear_static_faces()
    return update_preview_image(frame_number)


def slide_preview_image(frame_number: int = 0) -> gradio.update:
    if is_video(state_manager.get_item('target_path')):
        preview_vision_frame = normalize_frame_color(
            get_video_frame(state_manager.get_item('target_path'), frame_number))
        preview_vision_frame = resize_frame_resolution(preview_vision_frame, (1024, 1024))
        return gradio.update(value=preview_vision_frame)
    return gradio.update(value=None)


def update_preview_image(frame_number: int = 0) -> Tuple[gradio.update, gradio.update, gradio.update]:
    while process_manager.is_checking():
        sleep(0.5)

    # Initialize placeholders
    preview = gradio.update(value=None, visible=False)
    enable_button, disable_button = gradio.update(), gradio.update()

    try:
        conditional_append_reference_faces()
        reference_faces, reference_faces_2 = get_reference_faces() if 'reference' in state_manager.get_item(
            'face_selector_mode') else (None, None)
        source_face, source_face_2 = get_avg_faces()
        source_audio_frame = create_empty_audio_frame()
        source_audio_frame_2 = create_empty_audio_frame()

        if is_image(state_manager.get_item('target_path')):
            target_vision_frame = read_static_image(state_manager.get_item('target_path'))
            if target_vision_frame is not None:
                preview_vision_frame = process_preview_frame(
                    reference_faces, reference_faces_2, source_face, source_face_2,
                    source_audio_frame, source_audio_frame_2, target_vision_frame
                )
                preview_vision_frame = normalize_frame_color(preview_vision_frame)
                preview = gradio.update(value=preview_vision_frame, visible=True)

        elif is_video(state_manager.get_item('target_path')):
            temp_vision_frame = get_video_frame(state_manager.get_item('target_path'), frame_number)
            if temp_vision_frame is not None:
                preview_vision_frame = process_preview_frame(
                    reference_faces, reference_faces_2, source_face, source_face_2,
                    source_audio_frame, source_audio_frame_2, temp_vision_frame
                )
                preview_vision_frame = normalize_frame_color(preview_vision_frame)
                preview = gradio.update(value=preview_vision_frame, visible=True)

        # Update mask buttons
        enable_button, disable_button = update_mask_buttons(frame_number)

    except Exception as e:
        print(f"Error in update_preview_image: {e}")
        traceback.print_exc()

    return preview, enable_button, disable_button


# def update_preview_image_old(frame_number: int = 0) -> Tuple[gradio.update, gradio.update, gradio.update]:
#     global CURRENT_PREVIEW_FRAME_NUMBER
#     while process_manager.is_checking():
#         sleep(0.5)
#     conditional_append_reference_faces()
#     reference_faces = get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else None
#     source_face, source_face_2 = get_avg_faces()
#     source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
#     if source_audio_path and facefusion.globals.output_video_fps:
#         source_audio_frame = get_audio_frame(source_audio_path, facefusion.globals.output_video_fps, frame_number)
#     else:
#         source_audio_frame = None
#
#     enable_button, disable_button = update_mask_buttons(frame_number)
#     reference_faces, reference_faces_2 = (
#         get_reference_faces() if 'reference' in facefusion.globals.face_selector_mode else (None, None))
#     if is_image(state_manager.get_item('target_path')):
#         target_frame = read_static_image(state_manager.get_item('target_path'))
#         preview_frame = process_preview_frame(reference_faces, reference_faces_2, source_face, source_face_2,
#                                               source_audio_frame, target_frame, -1)
#         preview_frame = normalize_frame_color(preview_frame)
#         return gradio.update(value=preview_frame, visible=True), enable_button, disable_button
#     if is_video(state_manager.get_item('target_path')):
#         temp_frame = get_video_frame(state_manager.get_item('target_path'), frame_number)
#         try:
#             preview_frame = process_preview_frame(reference_faces, reference_faces_2, source_face, source_face_2,
#                                                   source_audio_frame, temp_frame, frame_number)
#             preview_frame = normalize_frame_color(preview_frame)
#         except Exception as e:
#             print("Error processing preview frame: ", e)
#             traceback.print_exc()
#             preview_frame = temp_frame
#         return gradio.update(value=preview_frame, visible=True), enable_button, disable_button
#     return gradio.update(value=None, visible=True), enable_button, disable_button


def preview_back(reference_frame_number: int = 0) -> gradio.update:
    frames_per_second = int(detect_video_fps(state_manager.get_item('target_path')))
    reference_frame_number = max(0, reference_frame_number - frames_per_second)
    preview, enable_btn, disable_btn = update_preview_image(reference_frame_number)
    return gradio.update(value=reference_frame_number), preview, enable_btn, disable_btn


def preview_forward(reference_frame_number: int = 0) -> gradio.update:
    frames_per_second = int(detect_video_fps(state_manager.get_item('target_path')))
    reference_frame_number = min(reference_frame_number + frames_per_second,
                                 count_video_frame_total(state_manager.get_item('target_path')))
    preview, enable_btn, disable_btn = update_preview_image(reference_frame_number)
    return gradio.update(value=reference_frame_number), preview, enable_btn, disable_btn


def preview_back_five(reference_frame_number: int = 0) -> gradio.update:
    frames_per_second = int(detect_video_fps(state_manager.get_item('target_path'))) * 5
    reference_frame_number = max(0, reference_frame_number - frames_per_second)
    preview, enable_btn, disable_btn = update_preview_image(reference_frame_number)
    return gradio.update(value=reference_frame_number), preview, enable_btn, disable_btn


def preview_forward_five(reference_frame_number: int = 0) -> gradio.update:
    frames_per_second = int(detect_video_fps(state_manager.get_item('target_path'))) * 5
    reference_frame_number = min(reference_frame_number + frames_per_second,
                                 count_video_frame_total(state_manager.get_item('target_path')))
    preview, enable_btn, disable_btn = update_preview_image(reference_frame_number)
    return gradio.update(value=reference_frame_number), preview, enable_btn, disable_btn


def update_preview_frame_slider() -> gradio.update:
    if is_video(state_manager.get_item('target_path')):
        video_frame_total = count_video_frame_total(state_manager.get_item('target_path'))
        return gradio.update(maximum=video_frame_total, visible=True), gradio.update(visible=True), gradio.update(
            visible=True), gradio.update(visible=True), gradio.update(visible=True), gradio.update(visible=True)
    return gradio.update(value=None, maximum=None, visible=False), gradio.update(visible=False), gradio.update(
        visible=False), gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=False)


def process_preview_frame(reference_faces: FaceSet, reference_faces_2: FaceSet, source_face: Face, source_face_2: Face,
                          source_audio_frame: AudioFrame, source_audio_frame_2: AudioFrame,
                          target_vision_frame: VisionFrame,
                          frame_number=-1) -> VisionFrame:
    with frame_processing_lock:
        print("Processing preview frame")
        target_vision_frame = resize_frame_resolution(target_vision_frame, (640, 640))
        if analyse_frame(target_vision_frame):
            return cv2.GaussianBlur(target_vision_frame, (99, 99), 0)
        global_processors = state_manager.get_item('processors')
        priority_order = ['face_swapper', 'style_changer', 'lip_syncer', 'face_enhancer', 'frame_enhancer',
                          'face_debugger']

        # Sort global_processors based on the priority_order
        global_processors = sorted(
            global_processors,
            key=lambda fp: priority_order.index(fp) if fp in priority_order else len(priority_order)
        )
        source_frame = target_vision_frame.copy()

        for frame_processor in global_processors:
            try:
                print(f"Processing with frame processor {frame_processor}")
                frame_processor_module = load_processor_module(frame_processor)
                if frame_processor_module.pre_process('preview'):
                    target_vision_frame = frame_processor_module.process_frame({
                        'reference_faces': reference_faces,
                        'reference_faces_2': reference_faces_2,
                        'source_face': source_face,
                        'source_face_2': source_face_2,
                        'source_audio_frame': source_audio_frame,
                        'source_audio_frame_2': source_audio_frame_2,
                        'target_vision_frame': target_vision_frame,
                        'target_frame_number': frame_number,
                        'source_frame': source_frame,
                    })
                    print("Do the damned thang...")
            except Exception as e:
                print(f"Error processing with frame processor {frame_processor}: {e}")
                traceback.print_exc()
        return target_vision_frame
