from typing import Optional, Tuple, List

import gradio

import facefusion.choices
from facefusion import wording, state_manager
from facefusion.common_helper import calc_int_step, calc_float_step
from facefusion.typing import FaceMaskType, FaceMaskRegion
from facefusion.uis.core import register_ui_component, get_ui_component, get_ui_components

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

    has_box_mask = 'box' in state_manager.get_item('face_mask_types')
    has_region_mask = 'region' in state_manager.get_item('face_mask_types')
    FACE_MASK_TYPES_CHECKBOX_GROUP = gradio.CheckboxGroup(
        label=wording.get('uis.face_mask_types_checkbox_group'),
        choices=facefusion.choices.face_mask_types,
        value=state_manager.get_item('face_mask_types')
    )
    FACE_MASK_REGIONS_CHECKBOX_GROUP = gradio.CheckboxGroup(
        label=wording.get('uis.face_mask_regions_checkbox_group'),
        choices=facefusion.choices.face_mask_regions,
        value=state_manager.get_item('face_mask_regions'),
        visible=has_region_mask
    )
    FACE_MASK_BLUR_SLIDER = gradio.Slider(
        label=wording.get('uis.face_mask_blur_slider'),
        step=calc_float_step(facefusion.choices.face_mask_blur_range),
        minimum=facefusion.choices.face_mask_blur_range[0],
        maximum=facefusion.choices.face_mask_blur_range[-1],
        value=state_manager.get_item('face_mask_blur'),
        visible=has_box_mask
    )
    with gradio.Group():
        with gradio.Row():
            MASK_DISABLE_BUTTON = gradio.Button(value="Disable Padding", variant="secondary", visible=False,
                                                elem_classes=["maskBtn"])
            MASK_ENABLE_BUTTON = gradio.Button(value="Enable Padding", variant="primary", visible=True,
                                               elem_classes=["maskBtn"])
            MASK_CLEAR_BUTTON = gradio.Button(value="Clear Markers", elem_classes=["maskBtn"])
            BOTTOM_MASK_POSITIONS = gradio.HTML(value=generate_frame_html(True),
                                                elem_id="bottom_mask_positions")
        with gradio.Row():
            FACE_MASK_PADDING_TOP_SLIDER = gradio.Slider(
                label=wording.get('uis.face_mask_padding_top_slider'),
                step=calc_int_step(facefusion.choices.face_mask_padding_range),
                minimum=facefusion.choices.face_mask_padding_range[0],
                maximum=facefusion.choices.face_mask_padding_range[-1],
                value=state_manager.get_item('face_mask_padding')[0],
                visible=has_box_mask
            )
            FACE_MASK_PADDING_RIGHT_SLIDER = gradio.Slider(
                label=wording.get('uis.face_mask_padding_right_slider'),
                step=calc_int_step(facefusion.choices.face_mask_padding_range),
                minimum=facefusion.choices.face_mask_padding_range[0],
                maximum=facefusion.choices.face_mask_padding_range[-1],
                value=state_manager.get_item('face_mask_padding')[1],
                visible=has_box_mask
            )
        with gradio.Row():
            FACE_MASK_PADDING_BOTTOM_SLIDER = gradio.Slider(
                label=wording.get('uis.face_mask_padding_bottom_slider'),
                step=calc_int_step(facefusion.choices.face_mask_padding_range),
                minimum=facefusion.choices.face_mask_padding_range[0],
                maximum=facefusion.choices.face_mask_padding_range[-1],
                value=state_manager.get_item('face_mask_padding')[2],
                visible=has_box_mask
            )
            FACE_MASK_PADDING_LEFT_SLIDER = gradio.Slider(
                label=wording.get('uis.face_mask_padding_left_slider'),
                step=calc_int_step(facefusion.choices.face_mask_padding_range),
                minimum=facefusion.choices.face_mask_padding_range[0],
                maximum=facefusion.choices.face_mask_padding_range[-1],
                value=state_manager.get_item('face_mask_padding')[3],
                visible=has_box_mask
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


def update_face_mask_types(face_mask_types: List[FaceMaskType]) -> Tuple[
    gradio.update, gradio.update, gradio.update, gradio.update, gradio.update, gradio.update, gradio.update]:
    face_mask_types = face_mask_types or facefusion.choices.face_mask_types
    state_manager.set_item('face_mask_types', face_mask_types)
    has_box_mask = 'box' in face_mask_types
    has_region_mask = 'region' in face_mask_types
    return gradio.update(value=state_manager.get_item('face_mask_types')), gradio.update(
        visible=has_region_mask), gradio.update(visible=has_box_mask), gradio.update(
        visible=has_box_mask), gradio.update(visible=has_box_mask), gradio.update(visible=has_box_mask), gradio.update(
        visible=has_box_mask)


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
        print(f"Current frame {frame_number} is not within a padding interval, showing enable_mask_button")
        return gradio.update(visible=True), gradio.update(visible=False)
    print(f"Current frame {frame_number} is within a padding interval, showing disable_mask_button")
    # We are currently enabled, so show the disable button, hide the enable button
    return gradio.update(visible=False), gradio.update(visible=True)
