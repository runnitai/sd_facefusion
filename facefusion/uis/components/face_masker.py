from typing import Optional, Tuple, List
import gradio

import facefusion.globals
import facefusion.choices
from facefusion import wording
from facefusion.typing import FaceMaskType, FaceMaskRegion
from facefusion.uis.core import register_ui_component, get_ui_component

FACE_MASK_TYPES_CHECKBOX_GROUP: Optional[gradio.CheckboxGroup] = None
FACE_MASK_BLUR_SLIDER: Optional[gradio.Slider] = None
FACE_MASK_BOX_GROUP: Optional[gradio.Group] = None
FACE_MASK_REGION_GROUP: Optional[gradio.Group] = None
FACE_MASK_PADDING_TOP_SLIDER: Optional[gradio.Slider] = None
FACE_MASK_PADDING_RIGHT_SLIDER: Optional[gradio.Slider] = None
FACE_MASK_PADDING_BOTTOM_SLIDER: Optional[gradio.Slider] = None
FACE_MASK_PADDING_LEFT_SLIDER: Optional[gradio.Slider] = None
FACE_MASK_REGION_CHECKBOX_GROUP: Optional[gradio.CheckboxGroup] = None
MASK_DISABLE_BUTTON: Optional[gradio.Button] = None
MASK_ENABLE_BUTTON: Optional[gradio.Button] = None
MASK_CLEAR_BUTTON: Optional[gradio.Button] = None
BOTTOM_MASK_POSITIONS: Optional[gradio.HTML] = None


def render() -> None:
    global FACE_MASK_TYPES_CHECKBOX_GROUP
    global FACE_MASK_BLUR_SLIDER
    global FACE_MASK_BOX_GROUP
    global FACE_MASK_REGION_GROUP
    global FACE_MASK_PADDING_TOP_SLIDER
    global FACE_MASK_PADDING_RIGHT_SLIDER
    global FACE_MASK_PADDING_BOTTOM_SLIDER
    global FACE_MASK_PADDING_LEFT_SLIDER
    global FACE_MASK_REGION_CHECKBOX_GROUP
    global MASK_DISABLE_BUTTON
    global MASK_ENABLE_BUTTON
    global MASK_CLEAR_BUTTON
    global BOTTOM_MASK_POSITIONS

    has_box_mask = 'box' in facefusion.globals.face_mask_types
    has_region_mask = 'region' in facefusion.globals.face_mask_types
    FACE_MASK_TYPES_CHECKBOX_GROUP = gradio.CheckboxGroup(
        label=wording.get('uis.face_mask_types_checkbox_group'),
        choices=facefusion.choices.face_mask_types,
        value=facefusion.globals.face_mask_types
    )
    with gradio.Group(visible=has_box_mask) as FACE_MASK_BOX_GROUP:
        FACE_MASK_BLUR_SLIDER = gradio.Slider(
            label=wording.get('uis.face_mask_blur_slider'),
            step=facefusion.choices.face_mask_blur_range[1] - facefusion.choices.face_mask_blur_range[0],
            minimum=facefusion.choices.face_mask_blur_range[0],
            maximum=facefusion.choices.face_mask_blur_range[-1],
            value=facefusion.globals.face_mask_blur
        )
        with gradio.Row():
            MASK_DISABLE_BUTTON = gradio.Button(value="Disable Mask Padding", variant="secondary", visible=False)
            MASK_ENABLE_BUTTON = gradio.Button(value="Enable Mask Padding", variant="primary", visible=True)
            MASK_CLEAR_BUTTON = gradio.Button(value="Clear Markers")
            BOTTOM_MASK_POSITIONS = gradio.HTML(value=generate_frame_html(True))
        with gradio.Row():
            FACE_MASK_PADDING_TOP_SLIDER = gradio.Slider(
                label=wording.get('uis.face_mask_padding_top_slider'),
                step=facefusion.choices.face_mask_padding_range[1] - facefusion.choices.face_mask_padding_range[0],
                minimum=facefusion.choices.face_mask_padding_range[0],
                maximum=facefusion.choices.face_mask_padding_range[-1],
                value=facefusion.globals.face_mask_padding[0]
            )
            FACE_MASK_PADDING_RIGHT_SLIDER = gradio.Slider(
                label=wording.get('uis.face_mask_padding_right_slider'),
                step=facefusion.choices.face_mask_padding_range[1] - facefusion.choices.face_mask_padding_range[0],
                minimum=facefusion.choices.face_mask_padding_range[0],
                maximum=facefusion.choices.face_mask_padding_range[-1],
                value=facefusion.globals.face_mask_padding[1]
            )
        with gradio.Row():
            FACE_MASK_PADDING_BOTTOM_SLIDER = gradio.Slider(
                label=wording.get('uis.face_mask_padding_bottom_slider'),
                step=facefusion.choices.face_mask_padding_range[1] - facefusion.choices.face_mask_padding_range[0],
                minimum=facefusion.choices.face_mask_padding_range[0],
                maximum=facefusion.choices.face_mask_padding_range[-1],
                value=facefusion.globals.face_mask_padding[2]
            )
            FACE_MASK_PADDING_LEFT_SLIDER = gradio.Slider(
                label=wording.get('uis.face_mask_padding_left_slider'),
                step=facefusion.choices.face_mask_padding_range[1] - facefusion.choices.face_mask_padding_range[0],
                minimum=facefusion.choices.face_mask_padding_range[0],
                maximum=facefusion.choices.face_mask_padding_range[-1],
                value=facefusion.globals.face_mask_padding[3]
            )
    with gradio.Row():
        FACE_MASK_REGION_CHECKBOX_GROUP = gradio.CheckboxGroup(
            label=wording.get('uis.face_mask_region_checkbox_group'),
            choices=facefusion.choices.face_mask_regions,
            value=facefusion.globals.face_mask_regions,
            visible=has_region_mask
        )
    register_ui_component('face_mask_types_checkbox_group', FACE_MASK_TYPES_CHECKBOX_GROUP)
    register_ui_component('face_mask_blur_slider', FACE_MASK_BLUR_SLIDER)
    register_ui_component('face_mask_padding_top_slider', FACE_MASK_PADDING_TOP_SLIDER)
    register_ui_component('face_mask_padding_right_slider', FACE_MASK_PADDING_RIGHT_SLIDER)
    register_ui_component('face_mask_padding_bottom_slider', FACE_MASK_PADDING_BOTTOM_SLIDER)
    register_ui_component('face_mask_padding_left_slider', FACE_MASK_PADDING_LEFT_SLIDER)
    register_ui_component('face_mask_region_checkbox_group', FACE_MASK_REGION_CHECKBOX_GROUP)
    register_ui_component("bottom_mask_positions", BOTTOM_MASK_POSITIONS)
    register_ui_component("mask_disable_button", MASK_DISABLE_BUTTON)
    register_ui_component("mask_enable_button", MASK_ENABLE_BUTTON)
    register_ui_component("mask_clear_button", MASK_CLEAR_BUTTON)
    register_ui_component("face_mask_box_group", FACE_MASK_BOX_GROUP)


def listen() -> None:
    FACE_MASK_TYPES_CHECKBOX_GROUP.change(update_face_mask_type, inputs=FACE_MASK_TYPES_CHECKBOX_GROUP,
                                          outputs=[FACE_MASK_TYPES_CHECKBOX_GROUP, FACE_MASK_BOX_GROUP,
                                                   FACE_MASK_REGION_CHECKBOX_GROUP])
    FACE_MASK_BLUR_SLIDER.change(update_face_mask_blur, inputs=FACE_MASK_BLUR_SLIDER)
    FACE_MASK_REGION_CHECKBOX_GROUP.change(update_face_mask_regions, inputs=FACE_MASK_REGION_CHECKBOX_GROUP,
                                           outputs=FACE_MASK_REGION_CHECKBOX_GROUP)
    face_mask_padding_sliders = [FACE_MASK_PADDING_TOP_SLIDER, FACE_MASK_PADDING_RIGHT_SLIDER,
                                 FACE_MASK_PADDING_BOTTOM_SLIDER, FACE_MASK_PADDING_LEFT_SLIDER]
    preview_frame_slider = get_ui_component("preview_frame_slider")
    mask_elements = [BOTTOM_MASK_POSITIONS, MASK_ENABLE_BUTTON, MASK_DISABLE_BUTTON]
    MASK_DISABLE_BUTTON.click(set_disable_mask_time, inputs=preview_frame_slider, outputs=mask_elements)
    MASK_ENABLE_BUTTON.click(set_enable_mask_time, inputs=preview_frame_slider, outputs=mask_elements)
    MASK_CLEAR_BUTTON.click(clear_mask_times, outputs=mask_elements)
    for face_mask_padding_slider in face_mask_padding_sliders:
        face_mask_padding_slider.change(update_face_mask_padding, inputs=face_mask_padding_sliders)


def generate_frame_html(return_value: bool = False) -> gradio.update:
    start_frames = facefusion.globals.mask_enabled_times
    end_frames = facefusion.globals.mask_disabled_times
    start_frame_string = ', '.join([str(start_frame) for start_frame in start_frames])
    end_frame_string = ', '.join([str(end_frame) for end_frame in end_frames])
    full_string = f"<p>Start Frames: {start_frame_string}</p><p>End Frames: {end_frame_string}</p>"
    if return_value:
        return full_string
    return gradio.update(value=full_string)


def set_disable_mask_time(preview_frame_slider: gradio.Slider) -> gradio.update:
    disabled_times = facefusion.globals.mask_disabled_times
    enabled_times = facefusion.globals.mask_enabled_times
    current_frame = preview_frame_slider
    if current_frame not in disabled_times:
        disabled_times.append(current_frame)
        disabled_times.sort()
    if current_frame in enabled_times:
        enabled_times.remove(current_frame)
        enabled_times.sort()
    facefusion.globals.mask_disabled_times = disabled_times
    facefusion.globals.mask_enabled_times = enabled_times
    show_enable_btn, show_disable_btn = update_mask_buttons(current_frame)
    return generate_frame_html(), show_enable_btn, show_disable_btn


def set_enable_mask_time(preview_frame_slider: gradio.Slider) -> gradio.update:
    disabled_times = facefusion.globals.mask_disabled_times
    enabled_times = facefusion.globals.mask_enabled_times
    current_frame = preview_frame_slider
    if current_frame not in enabled_times:
        enabled_times.append(current_frame)
        enabled_times.sort()
    if current_frame in disabled_times:
        disabled_times.remove(current_frame)
        disabled_times.sort()
    facefusion.globals.mask_disabled_times = disabled_times
    facefusion.globals.mask_enabled_times = enabled_times
    show_enable_btn, show_disable_btn = update_mask_buttons(current_frame)
    return generate_frame_html(), show_enable_btn, show_disable_btn


def clear_mask_times() -> gradio.update:
    facefusion.globals.mask_disabled_times = [0]
    facefusion.globals.mask_enabled_times = []
    show_enable_btn, show_disable_btn = update_mask_buttons(0)
    return generate_frame_html(), show_enable_btn, show_disable_btn


def update_mask_buttons(frame_number) -> (gradio.update, gradio.update):
    """Returns a tuple of (show_enable_btn, show_disable_btn)"""""
    if frame_number == -1:
        return gradio.update(visible=True), gradio.update(visible=False)

    disabled_times = facefusion.globals.mask_disabled_times
    enabled_times = facefusion.globals.mask_enabled_times

    # Get the latest start frame that is less than or equal to the current frame number
    latest_disabled_frame = max([frame for frame in disabled_times if frame <= frame_number], default=None)

    # Get the latest end frame that is less than or equal to the current frame number
    latest_enabled_frame = max([frame for frame in enabled_times if frame <= frame_number], default=None)

    # Determine if the current frame number is within a padding interval
    if latest_disabled_frame is not None and (
            latest_enabled_frame is None or latest_disabled_frame > latest_enabled_frame):
        # We are currently disabled, so show the enable button, hide the disable button
        print(f"Current frame {frame_number} is within a padding interval, showing enable_mask_button")
        return gradio.update(visible=True), gradio.update(visible=False)
    print(f"Current frame {frame_number} is not within a padding interval, showing disable_mask_button")
    # We are currently enabled, so show the disable button, hide the enable button
    return gradio.update(visible=False), gradio.update(visible=True)


def update_face_mask_type(face_mask_types: List[FaceMaskType]) -> Tuple[gradio.update, gradio.update, gradio.update]:
    if not face_mask_types:
        face_mask_types = facefusion.choices.face_mask_types
    facefusion.globals.face_mask_types = face_mask_types
    has_box_mask = 'box' in face_mask_types
    has_region_mask = 'region' in face_mask_types
    return gradio.update(value=face_mask_types), gradio.update(visible=has_box_mask), gradio.update(
        visible=has_region_mask)


def update_face_mask_blur(face_mask_blur: float) -> None:
    facefusion.globals.face_mask_blur = face_mask_blur


def update_face_mask_padding(face_mask_padding_top: int, face_mask_padding_right: int, face_mask_padding_bottom: int,
                             face_mask_padding_left: int) -> None:
    facefusion.globals.face_mask_padding = (
        face_mask_padding_top, face_mask_padding_right, face_mask_padding_bottom, face_mask_padding_left)


def update_face_mask_regions(face_mask_regions: List[FaceMaskRegion]) -> None:
    if not face_mask_regions:
        face_mask_regions = facefusion.choices.face_mask_regions
    facefusion.globals.face_mask_regions = face_mask_regions
    return gradio.update(value=face_mask_regions)
