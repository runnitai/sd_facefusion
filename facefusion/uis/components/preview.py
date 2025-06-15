import threading
import traceback
from datetime import datetime
from time import sleep
from typing import Any, Dict, Optional, Tuple

import cv2
import gradio

from facefusion import wording, process_manager, state_manager
from facefusion.audio import get_audio_frame
from facefusion.common_helper import get_first
from facefusion.face_analyser import get_average_faces
from facefusion.face_store import clear_static_faces, get_reference_faces, clear_reference_faces
from facefusion.filesystem import is_video, is_image, filter_audio_paths
from facefusion.processors.core import get_processors_modules
from facefusion.typing import Face, VisionFrame
from facefusion.uis.components.face_masker import update_mask_buttons
from facefusion.uis.core import get_ui_component, register_ui_component, get_ui_components
from facefusion.vision import get_video_frame, count_video_frame_total, normalize_frame_color, \
    read_static_image, detect_video_fps, resize_frame_resolution
from facefusion.workers.classes.content_analyser import ContentAnalyser

PREVIEW_IMAGE: Optional[gradio.Image] = None
PREVIEW_FRAME_SLIDER: Optional[gradio.Slider] = None
PREVIEW_FRAME_BACK_BUTTON: Optional[gradio.Button] = None
PREVIEW_FRAME_FORWARD_BUTTON: Optional[gradio.Button] = None
PREVIEW_FRAME_BACK_FIVE_BUTTON: Optional[gradio.Button] = None
PREVIEW_FRAME_FORWARD_FIVE_BUTTON: Optional[gradio.Button] = None
PREVIEW_FRAME_ROW: Optional[gradio.Row] = None

# Enhanced timeline controls
PLAY_PAUSE_BUTTON: Optional[gradio.Button] = None
FRAME_INFO_TEXT: Optional[gradio.Textbox] = None
PLAYBACK_SPEED_SLIDER: Optional[gradio.Slider] = None
TIMELINE_PROGRESS: Optional[gradio.HTML] = None

CURRENT_PREVIEW_FRAME_NUMBER = -1
AVG_FACE_1 = None
AVG_FACE_2 = None
SOURCE_FRAMES_1 = []
SOURCE_FRAMES_2 = []

frame_processing_lock = threading.Lock()
is_playing = False
playback_thread = None


def render() -> None:
    global PREVIEW_IMAGE
    global PREVIEW_FRAME_SLIDER
    global PREVIEW_FRAME_BACK_BUTTON
    global PREVIEW_FRAME_FORWARD_BUTTON
    global PREVIEW_FRAME_BACK_FIVE_BUTTON
    global PREVIEW_FRAME_FORWARD_FIVE_BUTTON
    global PREVIEW_FRAME_ROW
    global PLAY_PAUSE_BUTTON
    global FRAME_INFO_TEXT
    global PLAYBACK_SPEED_SLIDER
    global TIMELINE_PROGRESS

    preview_image_args: Dict[str, Any] = \
        {
            'label': wording.get('uis.preview_image'),
            'interactive': False,
            'elem_id': 'ff_preview_image'
        }
    
    preview_frame_slider_args: Dict[str, Any] = \
        {
            'label': wording.get('uis.preview_frame_slider'),
            'step': 1,
            'minimum': 0,
            'maximum': 100,
            'visible': True,
            'elem_id': 'ff_timeline_slider'
        }
    
    source_faces = get_average_faces()
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
    
    # Enhanced preview image with better styling
    if is_image(target_path):
        target_frame = read_static_image(target_path)
        preview_frame = process_preview_frame(source_faces,
                                              source_audio_frame, source_audio_frame_2, target_frame, -1)
        preview_image_args['value'] = normalize_frame_color(preview_frame)

    if is_video(target_path):
        frame_number = 0
        temp_frame = get_video_frame(target_path, frame_number)
        preview_frame = process_preview_frame(source_faces,
                                              source_audio_frame, source_audio_frame_2, temp_frame, frame_number)
        preview_image_args['value'] = normalize_frame_color(preview_frame)
        preview_frame_slider_args['value'] = 0
        preview_frame_slider_args['maximum'] = count_video_frame_total(target_path)

    # Main preview image
    PREVIEW_IMAGE = gradio.Image(**preview_image_args)
    
    # Enhanced timeline controls
    with gradio.Group(elem_id="timeline_controls_group") as timeline_group:
        # Frame information display
        FRAME_INFO_TEXT = gradio.Textbox(
            label="Frame Info",
            value=get_frame_info_text(0, target_path),
            interactive=False,
            elem_id="frame_info_text"
        )
        
        # Timeline progress bar (HTML for custom styling)
        TIMELINE_PROGRESS = gradio.HTML(
            value=get_timeline_html(0, preview_frame_slider_args.get('maximum', 100)),
            elem_id="timeline_progress"
        )
        
        # Main timeline slider
        PREVIEW_FRAME_SLIDER = gradio.Slider(**preview_frame_slider_args)
        
        # Enhanced playback controls
        with gradio.Row(visible=is_video(state_manager.get_item('target_path'))) as PREVIEW_FRAME_ROW:
            # Playback speed control
            PLAYBACK_SPEED_SLIDER = gradio.Slider(
                label="Speed",
                minimum=0.25,
                maximum=2.0,
                step=0.25,
                value=1.0,
                elem_id="playback_speed_slider",
                scale=1
            )
            
            # Frame navigation buttons
            PREVIEW_FRAME_BACK_FIVE_BUTTON = gradio.Button(
                value="⏪",
                elem_id='ff_preview_frame_back_five_button',
                elem_classes=['ff_preview_frame_button'],
                visible=preview_frame_slider_args['visible'],
                scale=0
            )

            PREVIEW_FRAME_BACK_BUTTON = gradio.Button(
                value="⏮️",
                elem_id='ff_preview_frame_back_button', 
                elem_classes=['ff_preview_frame_button'],
                visible=preview_frame_slider_args['visible'],
                scale=0
            )
            
            # Play/Pause button
            PLAY_PAUSE_BUTTON = gradio.Button(
                value="▶️",
                elem_id='ff_play_pause_button',
                elem_classes=['ff_preview_frame_button', 'play_button'],
                visible=preview_frame_slider_args['visible'],
                scale=0
            )
            
            PREVIEW_FRAME_FORWARD_BUTTON = gradio.Button(
                value="⏭️",
                elem_id='ff_preview_frame_forward_button',
                elem_classes=['ff_preview_frame_button'],
                visible=preview_frame_slider_args['visible'],
                scale=0
            )
            
            PREVIEW_FRAME_FORWARD_FIVE_BUTTON = gradio.Button(
                value="⏩",
                elem_id='ff_preview_frame_forward_five_button',
                elem_classes=['ff_preview_frame_button'],
                visible=preview_frame_slider_args['visible'],
                scale=0
            )

    register_ui_component('preview_frame_slider', PREVIEW_FRAME_SLIDER)
    register_ui_component('preview_frame_back_button', PREVIEW_FRAME_BACK_BUTTON)
    register_ui_component('preview_frame_forward_button', PREVIEW_FRAME_FORWARD_BUTTON)
    register_ui_component('preview_frame_back_five_button', PREVIEW_FRAME_BACK_FIVE_BUTTON)
    register_ui_component('preview_frame_forward_five_button', PREVIEW_FRAME_FORWARD_FIVE_BUTTON)
    register_ui_component('preview_image', PREVIEW_IMAGE)
    register_ui_component('preview_frame_row', PREVIEW_FRAME_ROW)
    register_ui_component('play_pause_button', PLAY_PAUSE_BUTTON)
    register_ui_component('frame_info_text', FRAME_INFO_TEXT)
    register_ui_component('playback_speed_slider', PLAYBACK_SPEED_SLIDER)
    register_ui_component('timeline_progress', TIMELINE_PROGRESS)


def get_frame_info_text(frame_number: int, target_path: str) -> str:
    """Generate informative frame information text"""
    if not target_path:
        return "No target selected"
    
    if is_video(target_path):
        total_frames = count_video_frame_total(target_path)
        fps = detect_video_fps(target_path)
        current_time = frame_number / fps if fps > 0 else 0
        total_time = total_frames / fps if fps > 0 else 0
        
        return f"Frame {frame_number:,} / {total_frames:,} | Time: {format_time(current_time)} / {format_time(total_time)} | FPS: {fps:.2f}"
    else:
        return f"Static Image | Frame {frame_number}"


def format_time(seconds: float) -> str:
    """Format time in MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def get_timeline_html(current_frame: int, total_frames: int) -> str:
    """Generate HTML for custom timeline progress bar"""
    if total_frames <= 0:
        progress_percent = 0
    else:
        progress_percent = (current_frame / total_frames) * 100
    
    return f"""
    <div style="background: #e2e8f0; border-radius: 8px; height: 8px; margin: 10px 0; overflow: hidden; position: relative;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 100%; width: {progress_percent:.1f}%; transition: width 0.3s ease;"></div>
        <div style="position: absolute; top: -20px; left: {progress_percent:.1f}%; transform: translateX(-50%); font-size: 0.8rem; color: #4a5568;">
            {progress_percent:.1f}%
        </div>
    </div>
    """


def toggle_playback(current_frame: int) -> Tuple[gradio.update, gradio.update]:
    """Toggle between play and pause"""
    global is_playing, playback_thread
    
    if is_playing:
        # Pause playback
        is_playing = False
        if playback_thread and playback_thread.is_alive():
            playback_thread.join(timeout=1.0)
        return gradio.update(value="▶️"), gradio.update()
    else:
        # Start playback
        is_playing = True
        playback_thread = threading.Thread(target=playback_loop, args=(current_frame,))
        playback_thread.start()
        return gradio.update(value="⏸️"), gradio.update()


def playback_loop(start_frame: int):
    """Playback loop for video preview"""
    global is_playing
    target_path = state_manager.get_item('target_path')
    if not is_video(target_path):
        return
    
    total_frames = count_video_frame_total(target_path)
    fps = detect_video_fps(target_path)
    playback_speed = 1.0  # This should be linked to the speed slider
    
    current_frame = start_frame
    
    while is_playing and current_frame < total_frames:
        try:
            sleep(1.0 / (fps * playback_speed))
            current_frame += 1
        except:
            break
    
    is_playing = False


def listen() -> None:
    mask_disable_button = get_ui_component('mask_disable_button')
    mask_enable_button = get_ui_component('mask_enable_button')
    mask_clear = get_ui_component('mask_clear_button')
    all_update_elements = [PREVIEW_IMAGE, mask_enable_button, mask_disable_button]
    more_elements = [PREVIEW_FRAME_SLIDER, FRAME_INFO_TEXT, TIMELINE_PROGRESS] + all_update_elements
    
    # Enhanced timeline controls
    PREVIEW_FRAME_BACK_BUTTON.click(preview_back, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements,
                                    show_progress='hidden').then(update_preview_image_enhanced, inputs=PREVIEW_FRAME_SLIDER,
                                                                 outputs=more_elements, show_progress='hidden')
    PREVIEW_FRAME_BACK_FIVE_BUTTON.click(preview_back_five, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements,
                                         show_progress='hidden').then(update_preview_image_enhanced, inputs=PREVIEW_FRAME_SLIDER,
                                                                      outputs=more_elements,
                                                                      show_progress='hidden')
    PREVIEW_FRAME_FORWARD_BUTTON.click(preview_forward, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements,
                                       show_progress='hidden').then(update_preview_image_enhanced, inputs=PREVIEW_FRAME_SLIDER,
                                                                    outputs=more_elements, show_progress='hidden')
    PREVIEW_FRAME_FORWARD_FIVE_BUTTON.click(preview_forward_five, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements,
                                            show_progress='hidden').then(update_preview_image_enhanced,
                                                                         inputs=PREVIEW_FRAME_SLIDER,
                                                                         outputs=more_elements,
                                                                         show_progress='hidden')
    
    # Play/Pause functionality
    PLAY_PAUSE_BUTTON.click(toggle_playback, inputs=PREVIEW_FRAME_SLIDER, 
                           outputs=[PLAY_PAUSE_BUTTON, PREVIEW_FRAME_SLIDER], show_progress='hidden')
    
    # Enhanced slider updates
    PREVIEW_FRAME_SLIDER.release(update_preview_image_enhanced, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements,
                                 show_progress='hidden')
    
    # Existing functionality
    if mask_disable_button:
        mask_disable_button.click(update_preview_image_enhanced, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements,
                                  show_progress='hidden')
    if mask_enable_button:
        mask_enable_button.click(update_preview_image_enhanced, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements,
                                 show_progress='hidden')
    if mask_clear:
        mask_clear.click(update_preview_image_enhanced, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements,
                         show_progress='hidden')
    
    for ui_component in get_ui_components(
            [
                'source_audio',
                'source_image',
                'source_image_2',
                'target_image',
                'target_video',
                'style_transfer_images',
            ]):
        for method in ['upload', 'change', 'clear']:
            getattr(ui_component, method)(update_preview_image_enhanced, inputs=PREVIEW_FRAME_SLIDER,
                                          outputs=more_elements,
                                          show_progress='hidden')

    for ui_component in get_ui_components(
            [
                'target_image',
                'target_video'
            ]):
        for method in ['upload', 'change', 'clear']:
            getattr(ui_component, method)(update_preview_frame_slider,
                                          outputs=[PREVIEW_FRAME_SLIDER, PREVIEW_FRAME_ROW])
                                          
    for ui_component in get_ui_components(
            [
                'face_debugger_items_checkbox_group',
                'frame_colorizer_size_dropdown',
                'face_mask_types_checkbox_group',
                'face_mask_regions_checkbox_group',
                'style_changer_target_radio',
                'style_changer_skip_head_checkbox',
            ]):
        ui_component.change(update_preview_image_enhanced, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements,
                            show_progress='hidden')
                            
    for ui_component in get_ui_components([
        'age_modifier_direction_slider',
        'expression_restorer_factor_slider',
        'face_editor_eyebrow_direction_slider',
        'face_editor_eye_gaze_horizontal_slider'
    ]):
        if ui_component:
            ui_component.change(update_preview_image_enhanced, inputs=PREVIEW_FRAME_SLIDER, outputs=more_elements,
                                show_progress='hidden')


def update_preview_image_enhanced(frame_number: int = 0) -> Tuple[gradio.update, gradio.update, gradio.update, gradio.update, gradio.update, gradio.update]:
    """Enhanced preview update with timeline information"""
    target_path = state_manager.get_item('target_path')
    
    # Update frame info
    frame_info = get_frame_info_text(frame_number, target_path)
    
    # Update timeline progress
    total_frames = count_video_frame_total(target_path) if is_video(target_path) else 1
    timeline_html = get_timeline_html(frame_number, total_frames)
    
    # Get the standard preview update
    preview_update, mask_enable_update, mask_disable_update = update_preview_image(frame_number)
    
    return (
        preview_update,
        mask_enable_update, 
        mask_disable_update,
        gradio.update(value=frame_number),  # Slider update
        gradio.update(value=frame_info),    # Frame info update
        gradio.update(value=timeline_html)  # Timeline progress update
    )


def clear_and_update_preview_image(frame_number: int = 0) -> gradio.update:
    global CURRENT_PREVIEW_FRAME_NUMBER
    CURRENT_PREVIEW_FRAME_NUMBER = -1
    # Only clear face-related caches, NOT source files
    clear_reference_faces()
    clear_static_faces()
    preview, _, _ = update_preview_image(frame_number)
    return preview


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
    preview = gradio.update(value=None)
    enable_button, disable_button = gradio.update(), gradio.update()

    try:
        # Get source faces
        source_faces = get_average_faces()

        # No need to process audio here - lip syncer will get the chunks itself
        audio_features = None
        audio_features_2 = None

        if is_image(state_manager.get_item('target_path')):
            target_vision_frame = read_static_image(state_manager.get_item('target_path'))
            if target_vision_frame is not None:
                preview_vision_frame = process_preview_frame(
                    source_faces,
                    audio_features, audio_features_2, target_vision_frame, frame_number
                )
                preview_vision_frame = normalize_frame_color(preview_vision_frame)
                preview = gradio.update(value=preview_vision_frame, visible=True)

        elif is_video(state_manager.get_item('target_path')):
            temp_vision_frame = get_video_frame(state_manager.get_item('target_path'), frame_number)
            if temp_vision_frame is not None:
                preview_vision_frame = process_preview_frame(
                    source_faces,
                    audio_features, audio_features_2, temp_vision_frame, frame_number
                )
                preview_vision_frame = normalize_frame_color(preview_vision_frame)
                preview = gradio.update(value=preview_vision_frame, visible=True)

        # Update mask buttons
        enable_button, disable_button = update_mask_buttons(frame_number)

    except Exception as e:
        print(f"Error in update_preview_image: {e}")
        traceback.print_exc()

    return preview, enable_button, disable_button


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
        return gradio.update(maximum=video_frame_total, visible=True), gradio.update(visible=True)
    return gradio.update(value=None, maximum=None, visible=False), gradio.update(visible=False)


def process_preview_frame(source_faces: Dict[int, Face],
                          audio_features, audio_features_2,
                          target_vision_frame: VisionFrame,
                          frame_number=0) -> VisionFrame:
    with frame_processing_lock:
        # Ensure frame_number is non-negative
        if frame_number < 0:
            frame_number = 0

        target_vision_frame = resize_frame_resolution(target_vision_frame, (640, 640))
        analyser = ContentAnalyser()
        if analyser.analyse_frame(target_vision_frame):
            return cv2.GaussianBlur(target_vision_frame, (99, 99), 0)
        global_processors = state_manager.get_item('processors')
        processors = get_processors_modules(global_processors)
        source_frame = target_vision_frame.copy()

        for frame_processor_module in processors:
            reference_faces = (
                get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else (
                    None, None))
            try:
                start_time = datetime.now()
                #frame_processor_module = load_processor_module(frame_processor)
                if frame_processor_module.pre_process('preview'):
                    target_vision_frame = frame_processor_module.process_frame({
                        'reference_faces': reference_faces,
                        'source_faces': source_faces,
                        'source_visual_frame': source_frame,
                        'frame_index': frame_number,  # Pass frame index for audio chunk selection
                        'target_vision_frame': target_vision_frame,
                        'target_frame_number': frame_number,
                        'source_vision_frame': source_frame,
                        'is_preview': True,
                    })
                    print(f"Processed with {frame_processor_module.display_name} in {datetime.now() - start_time}")
            except Exception as e:
                print(f"Error processing with frame processor {frame_processor_module.display_name}: {e}")
                traceback.print_exc()
        return target_vision_frame
