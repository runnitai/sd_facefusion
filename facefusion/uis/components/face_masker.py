from typing import Optional, Tuple, List
import os
import gradio

import facefusion.choices
from facefusion import wording, state_manager
from facefusion.common_helper import calc_int_step, calc_float_step
from facefusion.processors.core import get_processors_modules
from facefusion.typing import FaceMaskType, FaceMaskRegion
from facefusion.uis.core import register_ui_component, get_ui_component, get_ui_components
from facefusion.filesystem import resolve_relative_path

FACE_MASK_TYPES_CHECKBOX_GROUP: Optional[gradio.CheckboxGroup] = None
FACE_MASK_REGIONS_CHECKBOX_GROUP: Optional[gradio.CheckboxGroup] = None
FACE_MASK_BLUR_SLIDER: Optional[gradio.Slider] = None
FACE_MASK_PADDING_TOP_SLIDER: Optional[gradio.Slider] = None
FACE_MASK_PADDING_RIGHT_SLIDER: Optional[gradio.Slider] = None
FACE_MASK_PADDING_BOTTOM_SLIDER: Optional[gradio.Slider] = None
FACE_MASK_PADDING_LEFT_SLIDER: Optional[gradio.Slider] = None
MASK_DISABLE_BUTTON: Optional[gradio.Button] = None
MASK_ENABLE_BUTTON: Optional[gradio.Button] = None
MASK_CLEAR_BUTTON: Optional[gradio.Button] = None
BOTTOM_MASK_POSITIONS: Optional[gradio.HTML] = None
FACE_MASK_PADDING_GROUP: Optional[gradio.Group] = None
AUTO_PADDING_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
AUTO_PADDING_CONFIDENCE_SLIDER: Optional[gradio.Slider] = None
AUTO_PADDING_INTERSECTION_THRESHOLD_SLIDER: Optional[gradio.Slider] = None
AUTO_PADDING_GROUP: Optional[gradio.Group] = None
AUTO_PADDING_STATUS: Optional[gradio.HTML] = None


def find_yolo_models():
    from modules.paths_internal import models_path
    adetailer_path = os.path.join(models_path, "adetailer")
    
    models = []
    
    # Check adetailer path
    if os.path.exists(adetailer_path):
        for file in os.listdir(adetailer_path):
            if file.endswith('.pt'):
                models.append(os.path.join(adetailer_path, file))
            
    return models
    

def render() -> None:
    global FACE_MASK_TYPES_CHECKBOX_GROUP
    global FACE_MASK_REGIONS_CHECKBOX_GROUP
    global FACE_MASK_BLUR_SLIDER
    global FACE_MASK_PADDING_TOP_SLIDER
    global FACE_MASK_PADDING_RIGHT_SLIDER
    global FACE_MASK_PADDING_BOTTOM_SLIDER
    global FACE_MASK_PADDING_LEFT_SLIDER
    global MASK_DISABLE_BUTTON
    global MASK_ENABLE_BUTTON
    global MASK_CLEAR_BUTTON
    global BOTTOM_MASK_POSITIONS
    global FACE_MASK_PADDING_GROUP
    global AUTO_PADDING_MODEL_DROPDOWN
    global AUTO_PADDING_CONFIDENCE_SLIDER
    global AUTO_PADDING_INTERSECTION_THRESHOLD_SLIDER
    global AUTO_PADDING_GROUP
    global AUTO_PADDING_STATUS

    face_mask_types = state_manager.get_item('face_mask_types') or ['box']
    has_box_mask = 'box' in face_mask_types
    has_region_mask = 'region' in face_mask_types
    auto_padding_model = state_manager.get_item('auto_padding_model') or "None"
    has_auto_padding = auto_padding_model and auto_padding_model != "None"
    non_face_processors = ['frame_colorizer', 'frame_enhancer', 'style_transfer']
    # Make the group visible if any face processor is selected
    show_group = False
    for processor in state_manager.get_item('processors'):
        if processor not in non_face_processors:
            show_group = True
            break
    with gradio.Group(visible=show_group) as FACE_MASK_PADDING_GROUP:
        # Filter out 'custom' from face mask types since it's replaced by auto-padding
        available_mask_types = [t for t in facefusion.choices.face_mask_types if t != 'custom']
        current_mask_types = [t for t in face_mask_types if t != 'custom']
        
        FACE_MASK_TYPES_CHECKBOX_GROUP = gradio.CheckboxGroup(
            label=wording.get('uis.face_mask_types_checkbox_group'),
            choices=available_mask_types,
            value=current_mask_types
        )
        FACE_MASK_REGIONS_CHECKBOX_GROUP = gradio.CheckboxGroup(
            label=wording.get('uis.face_mask_regions_checkbox_group'),
            choices=facefusion.choices.face_mask_regions,
            value=state_manager.get_item('face_mask_regions') or [],
            visible=has_region_mask
        )
        FACE_MASK_BLUR_SLIDER = gradio.Slider(
            label=wording.get('uis.face_mask_blur_slider'),
            step=calc_float_step(facefusion.choices.face_mask_blur_range),
            minimum=facefusion.choices.face_mask_blur_range[0],
            maximum=facefusion.choices.face_mask_blur_range[-1],
            value=state_manager.get_item('face_mask_blur') or 0.3,
            visible=has_box_mask
        )
        # Auto-padding model selection - always visible
        with gradio.Group() as AUTO_PADDING_GROUP:
            yolo_models = find_yolo_models()
            model_names = ["None"] + [os.path.basename(m) for m in yolo_models]
            model_dict = dict(zip([os.path.basename(m) for m in yolo_models], yolo_models))
            
            AUTO_PADDING_MODEL_DROPDOWN = gradio.Dropdown(
                label="Auto Padding Model",
                choices=model_names,
                value=auto_padding_model,
                type="value"
            )
            
            AUTO_PADDING_CONFIDENCE_SLIDER = gradio.Slider(
                label="Detection Confidence",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=state_manager.get_item('auto_padding_confidence') if state_manager.get_item('auto_padding_confidence') is not None else 0.5,
                visible=False  # Will be controlled by change listener
            )
            AUTO_PADDING_INTERSECTION_THRESHOLD_SLIDER = gradio.Slider(
                label="Intersection Threshold (pixels)",
                minimum=0,
                maximum=200,
                step=5,
                value=state_manager.get_item('auto_padding_intersection_threshold') if state_manager.get_item('auto_padding_intersection_threshold') is not None else 50,
                visible=False  # Will be controlled by change listener
            )
            
            AUTO_PADDING_STATUS = gradio.HTML(
                value="<div style='text-align: center; padding: 10px; color: #888;'>AUTO PADDING OFF</div>" if not has_auto_padding else "<div style='text-align: center; padding: 10px; color: #4CAF50;'>AUTO PADDING ON</div>"
            )
        with gradio.Row():
            MASK_DISABLE_BUTTON = gradio.Button(value="Disable Padding", variant="secondary", visible=False,
                                                elem_classes=["maskBtn"])
            MASK_ENABLE_BUTTON = gradio.Button(value="Enable Padding", variant="primary", visible=False,  # Will be controlled by change listener
                                               elem_classes=["maskBtn"])
            MASK_CLEAR_BUTTON = gradio.Button(value="Clear Markers", elem_classes=["maskBtn"], visible=False)  # Will be controlled by change listener
        with gradio.Row():
            BOTTOM_MASK_POSITIONS = gradio.HTML(value=generate_frame_html(True),
                                                elem_id="bottom_mask_positions")
        with gradio.Row():
            FACE_MASK_PADDING_TOP_SLIDER = gradio.Slider(
                label=wording.get('uis.face_mask_padding_top_slider'),
                step=calc_int_step(facefusion.choices.face_mask_padding_range),
                minimum=facefusion.choices.face_mask_padding_range[0],
                maximum=facefusion.choices.face_mask_padding_range[-1],
                value=(state_manager.get_item('face_mask_padding') or (0, 0, 0, 0))[0]
            )
            FACE_MASK_PADDING_RIGHT_SLIDER = gradio.Slider(
                label=wording.get('uis.face_mask_padding_right_slider'),
                step=calc_int_step(facefusion.choices.face_mask_padding_range),
                minimum=facefusion.choices.face_mask_padding_range[0],
                maximum=facefusion.choices.face_mask_padding_range[-1],
                value=(state_manager.get_item('face_mask_padding') or (0, 0, 0, 0))[1]
            )
        with gradio.Row():
            FACE_MASK_PADDING_BOTTOM_SLIDER = gradio.Slider(
                label=wording.get('uis.face_mask_padding_bottom_slider'),
                step=calc_int_step(facefusion.choices.face_mask_padding_range),
                minimum=facefusion.choices.face_mask_padding_range[0],
                maximum=facefusion.choices.face_mask_padding_range[-1],
                value=(state_manager.get_item('face_mask_padding') or (0, 0, 0, 0))[2]
            )
            FACE_MASK_PADDING_LEFT_SLIDER = gradio.Slider(
                label=wording.get('uis.face_mask_padding_left_slider'),
                step=calc_int_step(facefusion.choices.face_mask_padding_range),
                minimum=facefusion.choices.face_mask_padding_range[0],
                maximum=facefusion.choices.face_mask_padding_range[-1],
                value=(state_manager.get_item('face_mask_padding') or (0, 0, 0, 0))[3]
            )
    register_ui_component('face_mask_types_checkbox_group', FACE_MASK_TYPES_CHECKBOX_GROUP)
    register_ui_component('face_mask_regions_checkbox_group', FACE_MASK_REGIONS_CHECKBOX_GROUP)
    register_ui_component('face_mask_blur_slider', FACE_MASK_BLUR_SLIDER)
    register_ui_component('face_mask_padding_top_slider', FACE_MASK_PADDING_TOP_SLIDER)
    register_ui_component('face_mask_padding_right_slider', FACE_MASK_PADDING_RIGHT_SLIDER)
    register_ui_component('face_mask_padding_bottom_slider', FACE_MASK_PADDING_BOTTOM_SLIDER)
    register_ui_component('face_mask_padding_left_slider', FACE_MASK_PADDING_LEFT_SLIDER)
    register_ui_component("bottom_mask_positions", BOTTOM_MASK_POSITIONS)
    register_ui_component("mask_disable_button", MASK_DISABLE_BUTTON)
    register_ui_component("mask_enable_button", MASK_ENABLE_BUTTON)
    register_ui_component("mask_clear_button", MASK_CLEAR_BUTTON)
    register_ui_component("auto_padding_model_dropdown", AUTO_PADDING_MODEL_DROPDOWN)
    register_ui_component("auto_padding_confidence_slider", AUTO_PADDING_CONFIDENCE_SLIDER)
    register_ui_component("auto_padding_intersection_threshold_slider", AUTO_PADDING_INTERSECTION_THRESHOLD_SLIDER)
    register_ui_component("auto_padding_status", AUTO_PADDING_STATUS)


def listen() -> None:
    FACE_MASK_TYPES_CHECKBOX_GROUP.change(update_face_mask_types, inputs=FACE_MASK_TYPES_CHECKBOX_GROUP,
                                          outputs=[FACE_MASK_TYPES_CHECKBOX_GROUP, FACE_MASK_REGIONS_CHECKBOX_GROUP,
                                                   FACE_MASK_BLUR_SLIDER, FACE_MASK_PADDING_TOP_SLIDER,
                                                   FACE_MASK_PADDING_RIGHT_SLIDER, FACE_MASK_PADDING_BOTTOM_SLIDER,
                                                   FACE_MASK_PADDING_LEFT_SLIDER])
    FACE_MASK_REGIONS_CHECKBOX_GROUP.change(update_face_mask_regions, inputs=FACE_MASK_REGIONS_CHECKBOX_GROUP,
                                            outputs=FACE_MASK_REGIONS_CHECKBOX_GROUP)
    FACE_MASK_BLUR_SLIDER.release(update_face_mask_blur, inputs=FACE_MASK_BLUR_SLIDER)
    face_mask_padding_sliders = [FACE_MASK_PADDING_TOP_SLIDER, FACE_MASK_PADDING_RIGHT_SLIDER,
                                 FACE_MASK_PADDING_BOTTOM_SLIDER, FACE_MASK_PADDING_LEFT_SLIDER]

    for face_mask_padding_slider in face_mask_padding_sliders:
        face_mask_padding_slider.release(update_face_mask_padding, inputs=face_mask_padding_sliders)

    AUTO_PADDING_MODEL_DROPDOWN.change(update_auto_padding_model_and_ui, inputs=AUTO_PADDING_MODEL_DROPDOWN,
                                       outputs=[AUTO_PADDING_STATUS, AUTO_PADDING_CONFIDENCE_SLIDER, AUTO_PADDING_INTERSECTION_THRESHOLD_SLIDER,
                                               MASK_ENABLE_BUTTON, MASK_DISABLE_BUTTON, MASK_CLEAR_BUTTON])
    AUTO_PADDING_CONFIDENCE_SLIDER.release(update_auto_padding_confidence, inputs=AUTO_PADDING_CONFIDENCE_SLIDER)
    AUTO_PADDING_INTERSECTION_THRESHOLD_SLIDER.release(update_auto_padding_intersection_threshold, inputs=AUTO_PADDING_INTERSECTION_THRESHOLD_SLIDER)
    
    # Update auto-padding status when settings change
    AUTO_PADDING_CONFIDENCE_SLIDER.release(update_auto_padding_status, outputs=[AUTO_PADDING_STATUS])
    AUTO_PADDING_INTERSECTION_THRESHOLD_SLIDER.release(update_auto_padding_status, outputs=[AUTO_PADDING_STATUS])

    preview_frame_slider = get_ui_component("preview_frame_slider")
    mask_elements = [BOTTOM_MASK_POSITIONS, MASK_ENABLE_BUTTON, MASK_DISABLE_BUTTON]

    MASK_DISABLE_BUTTON.click(
        lambda preview_frame: set_disable_mask_time(preview_frame),
        inputs=preview_frame_slider,
        outputs=mask_elements
    )
    MASK_ENABLE_BUTTON.click(
        lambda preview_frame: set_enable_mask_time(preview_frame),
        inputs=preview_frame_slider,
        outputs=mask_elements
    )
    MASK_CLEAR_BUTTON.click(
        lambda: clear_mask_times(),
        outputs=mask_elements
    )

    for ui_component in get_ui_components(
            [
                'target_image',
                'target_video'
            ]):
        for method in ['upload', 'change', 'clear']:
            getattr(ui_component, method)(clear_mask_times,
                                          outputs=mask_elements)

    for method in ['change', 'release']:
        getattr(preview_frame_slider, method)(update_mask_buttons,
                                              inputs=preview_frame_slider,
                                              outputs=[MASK_ENABLE_BUTTON, MASK_DISABLE_BUTTON])
        
    # Update auto-padding status when preview frame changes
    for method in ['change', 'release']:
        getattr(preview_frame_slider, method)(update_auto_padding_status,
                                              inputs=preview_frame_slider,
                                              outputs=[AUTO_PADDING_STATUS])

    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(
            toggle_group,
            inputs=processors_checkbox_group,
            outputs=[FACE_MASK_PADDING_GROUP]
        )


def toggle_group(processors: List[str]) -> gradio.update:
    all_processors = get_processors_modules()
    all_face_processor_names = [processor.display_name for processor in all_processors if processor.is_face_processor]
    # Make the group visible if any face processor is selected
    for processor in processors:
        if processor in all_face_processor_names:
            return gradio.update(visible=True)
    return gradio.update(visible=False)


def update_face_mask_types(face_mask_types: List[FaceMaskType]) -> Tuple[
    gradio.update, gradio.update, gradio.update, gradio.update, gradio.update, gradio.update, gradio.update]:
    # Filter out 'custom' from face mask types since it's replaced by auto-padding
    available_mask_types = [t for t in facefusion.choices.face_mask_types if t != 'custom']
    face_mask_types = [t for t in (face_mask_types or available_mask_types) if t != 'custom']
    state_manager.set_item('face_mask_types', face_mask_types)
    has_box_mask = 'box' in face_mask_types
    has_region_mask = 'region' in face_mask_types
    auto_padding_model = state_manager.get_item('auto_padding_model') or "None"
    has_auto_padding = auto_padding_model and auto_padding_model != "None"
    return (
        gradio.update(value=face_mask_types),
        gradio.update(visible=has_region_mask),
        gradio.update(visible=has_box_mask and not has_auto_padding),
        gradio.update(visible=has_box_mask and not has_auto_padding),
        gradio.update(visible=has_box_mask and not has_auto_padding),
        gradio.update(visible=has_box_mask and not has_auto_padding),
        gradio.update(visible=has_box_mask and not has_auto_padding)
    )


def update_face_mask_regions(face_mask_regions: List[FaceMaskRegion]) -> gradio.update:
    face_mask_regions = face_mask_regions or facefusion.choices.face_mask_regions
    state_manager.set_item('face_mask_regions', face_mask_regions)
    return gradio.update(value=state_manager.get_item('face_mask_regions'))


def update_face_mask_blur(face_mask_blur: float) -> None:
    state_manager.set_item('face_mask_blur', face_mask_blur)


def update_face_mask_padding(face_mask_padding_top: float, face_mask_padding_right: float,
                             face_mask_padding_bottom: float, face_mask_padding_left: float) -> None:
    face_mask_padding = (int(face_mask_padding_top), int(face_mask_padding_right), int(face_mask_padding_bottom),
                         int(face_mask_padding_left))
    state_manager.set_item('face_mask_padding', face_mask_padding)


def update_auto_padding_model(auto_padding_model: str) -> gradio.update:
    state_manager.set_item('auto_padding_model', auto_padding_model)
    # Use the centralized status update function
    return update_auto_padding_status()


def update_auto_padding_model_and_ui(auto_padding_model: str) -> Tuple[gradio.update, gradio.update, gradio.update, gradio.update, gradio.update, gradio.update]:
    """Update auto-padding model and toggle UI visibility"""
    state_manager.set_item('auto_padding_model', auto_padding_model)
    has_auto_padding = auto_padding_model and auto_padding_model != "None"
    
    # Update status
    status_update = update_auto_padding_status()
    
    # Show/hide auto-padding settings - show when model is selected
    auto_confidence_visible = gradio.update(visible=has_auto_padding)
    auto_threshold_visible = gradio.update(visible=has_auto_padding)
    
    # Hide/show manual mask timing buttons - show when "None" is selected
    manual_timing_visible = not has_auto_padding
    mask_enable_visible = gradio.update(visible=manual_timing_visible)
    mask_disable_visible = gradio.update(visible=manual_timing_visible)
    mask_clear_visible = gradio.update(visible=manual_timing_visible)
    
    return (status_update, auto_confidence_visible, auto_threshold_visible,
            mask_enable_visible, mask_disable_visible, mask_clear_visible)


def update_auto_padding_confidence(auto_padding_confidence: float) -> None:
    state_manager.set_item('auto_padding_confidence', auto_padding_confidence)


def update_auto_padding_intersection_threshold(auto_padding_intersection_threshold: int) -> None:
    state_manager.set_item('auto_padding_intersection_threshold', auto_padding_intersection_threshold)


def update_auto_padding_status(frame_number: int = 0) -> gradio.update:
    """Update auto-padding status based on current frame and settings"""
    auto_padding_model = state_manager.get_item('auto_padding_model') or "None"
    has_auto_padding = auto_padding_model and auto_padding_model != "None"
    
    if not has_auto_padding:
        status_text = "<div style='text-align: center; padding: 10px; color: #888;'>AUTO PADDING OFF</div>"
    else:
        # For now, just show that auto-padding is active
        # TODO: In the future, we could check if objects were actually detected in the current frame
        model_name = os.path.basename(auto_padding_model) if auto_padding_model != "None" else "Unknown"
        confidence = state_manager.get_item('auto_padding_confidence') or 0.5
        status_text = f"<div style='text-align: center; padding: 10px; color: #4CAF50;'>AUTO PADDING ON<br/><small>{model_name} (conf: {confidence})</small></div>"
    
    return gradio.update(value=status_text)


def generate_frame_html(return_value: bool = False) -> gradio.update:
    start_frames = state_manager.get_item('mask_enabled_times')
    end_frames = state_manager.get_item('mask_disabled_times')
    start_frame_string = ', '.join([str(start_frame) for start_frame in start_frames]) if start_frames else ''
    end_frame_string = ', '.join([str(end_frame) for end_frame in end_frames]) if end_frames else ''
    full_string = f"<p>Start Frames: {start_frame_string}</p><p>End Frames: {end_frame_string}</p>"
    if return_value:
        return full_string
    return gradio.update(value=full_string)


def set_disable_mask_time(preview_frame_slider: gradio.update) -> gradio.update:
    disabled_times = state_manager.get_item('mask_disabled_times')
    enabled_times = state_manager.get_item('mask_enabled_times')
    current_frame = preview_frame_slider
    if current_frame not in disabled_times:
        disabled_times.append(current_frame)
        disabled_times.sort()
    if current_frame in enabled_times:
        enabled_times.remove(current_frame)
        enabled_times.sort()
    state_manager.set_item('mask_disabled_times', disabled_times)
    state_manager.set_item('mask_enabled_times', enabled_times)
    show_enable_btn, show_disable_btn = update_mask_buttons(current_frame)
    return generate_frame_html(state_manager), show_enable_btn, show_disable_btn


def set_enable_mask_time(preview_frame_slider: gradio.update) -> gradio.update:
    disabled_times = state_manager.get_item('mask_disabled_times')
    enabled_times = state_manager.get_item('mask_enabled_times')
    current_frame = preview_frame_slider
    if current_frame not in enabled_times:
        enabled_times.append(current_frame)
        enabled_times.sort()
    if current_frame in disabled_times:
        disabled_times.remove(current_frame)
        disabled_times.sort()
    state_manager.set_item('mask_disabled_times', disabled_times)
    state_manager.set_item('mask_enabled_times', enabled_times)
    show_enable_btn, show_disable_btn = update_mask_buttons(current_frame)
    return generate_frame_html(state_manager), show_enable_btn, show_disable_btn


def clear_mask_times() -> (gradio.update, gradio.update, gradio.update):
    state_manager.set_item('mask_disabled_times', [0])
    state_manager.set_item('mask_enabled_times', [])
    show_enable_btn, show_disable_btn = update_mask_buttons(0)
    return generate_frame_html(state_manager), show_enable_btn, show_disable_btn


def update_mask_buttons(frame_number) -> (gradio.update, gradio.update):
    """Returns a tuple of (show_enable_btn, show_disable_btn)"""
    if frame_number == -1:
        return gradio.update(visible=True), gradio.update(visible=False)

    disabled_times = state_manager.get_item('mask_disabled_times')
    enabled_times = state_manager.get_item('mask_enabled_times')

    # Get the latest start frame that is less than or equal to the current frame number
    if disabled_times:
        latest_disabled_frame = max([frame for frame in disabled_times if frame <= frame_number], default=None)
    else:
        latest_disabled_frame = None

    # Get the latest end frame that is less than or equal to the current frame number
    if enabled_times:
        latest_enabled_frame = max([frame for frame in enabled_times if frame <= frame_number], default=None)
    else:
        latest_enabled_frame = None

    # Determine if the current frame number is within a padding interval
    if latest_disabled_frame is not None and (
            latest_enabled_frame is None or latest_disabled_frame > latest_enabled_frame):
        # We are currently disabled, so show the enable button, hide the disable button
        return gradio.update(visible=True), gradio.update(visible=False)
    # We are currently enabled, so show the disable button, hide the enable button
    return gradio.update(visible=False), gradio.update(visible=True)
