from typing import Optional, Sequence, Tuple, List

import gradio

import facefusion.choices
from facefusion import choices, face_detector, state_manager, wording
from facefusion.common_helper import calc_float_step, get_last
from facefusion.typing import Angle, FaceDetectorModel, Score
from facefusion.uis.core import register_ui_component, get_ui_component
from facefusion.uis.typing import ComponentOptions

FACE_DETECTOR_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_DETECTOR_SIZE_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_DETECTOR_ANGLES_CHECKBOX_GROUP: Optional[gradio.CheckboxGroup] = None
FACE_DETECTOR_SCORE_SLIDER: Optional[gradio.Slider] = None
FACE_DETECTOR_GROUP: Optional[gradio.Group] = None


def render() -> None:
    global FACE_DETECTOR_MODEL_DROPDOWN
    global FACE_DETECTOR_SIZE_DROPDOWN
    global FACE_DETECTOR_ANGLES_CHECKBOX_GROUP
    global FACE_DETECTOR_SCORE_SLIDER
    global FACE_DETECTOR_GROUP

    face_detector_size_dropdown_options: ComponentOptions = \
        {
            'label': wording.get('uis.face_detector_size_dropdown'),
            'value': state_manager.get_item('face_detector_size')
        }
    if state_manager.get_item('face_detector_size') in facefusion.choices.face_detector_set[
        state_manager.get_item('face_detector_model')]:
        face_detector_size_dropdown_options['choices'] = facefusion.choices.face_detector_set[
            state_manager.get_item('face_detector_model')]
    non_face_processors = ['frame_colorizer', 'frame_enhancer']
    # Make the group visible if any face processor is selected
    processors = state_manager.get_item('processors')
    show_group = False
    for processor in processors:
        if processor not in non_face_processors:
            show_group = True
    with gradio.Group(visible=show_group) as FACE_DETECTOR_GROUP:
        with gradio.Row():
            FACE_DETECTOR_MODEL_DROPDOWN = gradio.Dropdown(
                label=wording.get('uis.face_detector_model_dropdown'),
                choices=facefusion.choices.face_detector_set.keys(),
                value=state_manager.get_item('face_detector_model')
            )
            FACE_DETECTOR_SIZE_DROPDOWN = gradio.Dropdown(**face_detector_size_dropdown_options)
        FACE_DETECTOR_ANGLES_CHECKBOX_GROUP = gradio.CheckboxGroup(
            label=wording.get('uis.face_detector_angles_checkbox_group'),
            choices=facefusion.choices.face_detector_angles,
            value=state_manager.get_item('face_detector_angles')
        )
        FACE_DETECTOR_SCORE_SLIDER = gradio.Slider(
            label=wording.get('uis.face_detector_score_slider'),
            value=state_manager.get_item('face_detector_score'),
            step=calc_float_step(facefusion.choices.face_detector_score_range),
            minimum=facefusion.choices.face_detector_score_range[0],
            maximum=facefusion.choices.face_detector_score_range[-1]
        )
    register_ui_component('face_detector_model_dropdown', FACE_DETECTOR_MODEL_DROPDOWN)
    register_ui_component('face_detector_size_dropdown', FACE_DETECTOR_SIZE_DROPDOWN)
    register_ui_component('face_detector_angles_checkbox_group', FACE_DETECTOR_ANGLES_CHECKBOX_GROUP)
    register_ui_component('face_detector_score_slider', FACE_DETECTOR_SCORE_SLIDER)
    register_ui_component('face_detector_group', FACE_DETECTOR_GROUP)


def listen() -> None:
    FACE_DETECTOR_MODEL_DROPDOWN.change(update_face_detector_model, inputs=FACE_DETECTOR_MODEL_DROPDOWN,
                                        outputs=[FACE_DETECTOR_MODEL_DROPDOWN, FACE_DETECTOR_SIZE_DROPDOWN])
    FACE_DETECTOR_SIZE_DROPDOWN.change(update_face_detector_size, inputs=FACE_DETECTOR_SIZE_DROPDOWN)
    FACE_DETECTOR_ANGLES_CHECKBOX_GROUP.change(update_face_detector_angles, inputs=FACE_DETECTOR_ANGLES_CHECKBOX_GROUP,
                                               outputs=FACE_DETECTOR_ANGLES_CHECKBOX_GROUP)
    FACE_DETECTOR_SCORE_SLIDER.release(update_face_detector_score, inputs=FACE_DETECTOR_SCORE_SLIDER)
    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(
            toggle_group,
            inputs=processors_checkbox_group,
            outputs=[FACE_DETECTOR_GROUP]
        )


def toggle_group(processors: List[str]) -> gradio.update:
    non_face_processors = ['frame_colorizer', 'frame_enhancer']
    # Make the group visible if any face processor is selected
    for processor in processors:
        if processor not in non_face_processors:
            return gradio.update(visible=True)
    return gradio.update(visible=False)


def update_face_detector_model(face_detector_model: FaceDetectorModel) -> Tuple[gradio.update, gradio.update]:
    face_detector.clear_inference_pool()
    state_manager.set_item('face_detector_model', face_detector_model)

    if face_detector.pre_check():
        face_detector_size_choices = choices.face_detector_set.get(state_manager.get_item('face_detector_model'))
        state_manager.set_item('face_detector_size', get_last(face_detector_size_choices))
        return gradio.update(value=state_manager.get_item('face_detector_model')), gradio.update(
            value=state_manager.get_item('face_detector_size'), choices=face_detector_size_choices)
    return gradio.update(), gradio.update()


def update_face_detector_size(face_detector_size: str) -> None:
    state_manager.set_item('face_detector_size', face_detector_size)


def update_face_detector_angles(face_detector_angles: Sequence[Angle]) -> gradio.update:
    face_detector_angles = face_detector_angles or facefusion.choices.face_detector_angles
    state_manager.set_item('face_detector_angles', face_detector_angles)
    return gradio.update(value=state_manager.get_item('face_detector_angles'))


def update_face_detector_score(face_detector_score: Score) -> None:
    state_manager.set_item('face_detector_score', face_detector_score)
