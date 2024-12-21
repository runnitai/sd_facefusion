from typing import Optional, List

import gradio

import facefusion.choices
from facefusion import state_manager, wording
from facefusion.common_helper import calc_float_step
from facefusion.processors.core import get_processors_modules
from facefusion.typing import FaceLandmarkerModel, Score
from facefusion.uis.core import register_ui_component, get_ui_component
from facefusion.workers.classes.face_landmarker import FaceLandmarker

FACE_LANDMARKER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_LANDMARKER_SCORE_SLIDER: Optional[gradio.Slider] = None
FACE_LANDMARKER_GROUP: Optional[gradio.Group] = None


def render() -> None:
    global FACE_LANDMARKER_MODEL_DROPDOWN
    global FACE_LANDMARKER_SCORE_SLIDER
    global FACE_LANDMARKER_GROUP
    non_face_processors = ['frame_colorizer', 'frame_enhancer', 'style_transfer']
    processors = state_manager.get_item('processors')
    show_group = False
    for processor in processors:
        if processor not in non_face_processors:
            show_group = True
    with gradio.Group(visible=show_group) as FACE_LANDMARKER_GROUP:
        FACE_LANDMARKER_MODEL_DROPDOWN = gradio.Dropdown(
            label=wording.get('uis.face_landmarker_model_dropdown'),
            choices=facefusion.choices.face_landmarker_models,
            value=state_manager.get_item('face_landmarker_model')
        )
        FACE_LANDMARKER_SCORE_SLIDER = gradio.Slider(
            label=wording.get('uis.face_landmarker_score_slider'),
            value=state_manager.get_item('face_landmarker_score'),
            step=calc_float_step(facefusion.choices.face_landmarker_score_range),
            minimum=facefusion.choices.face_landmarker_score_range[0],
            maximum=facefusion.choices.face_landmarker_score_range[-1]
        )
    register_ui_component('face_landmarker_model_dropdown', FACE_LANDMARKER_MODEL_DROPDOWN)
    register_ui_component('face_landmarker_score_slider', FACE_LANDMARKER_SCORE_SLIDER)
    register_ui_component('face_landmarker_group', FACE_LANDMARKER_GROUP)


def listen() -> None:
    FACE_LANDMARKER_MODEL_DROPDOWN.change(update_face_landmarker_model, inputs=FACE_LANDMARKER_MODEL_DROPDOWN,
                                          outputs=FACE_LANDMARKER_MODEL_DROPDOWN)
    FACE_LANDMARKER_SCORE_SLIDER.release(update_face_landmarker_score, inputs=FACE_LANDMARKER_SCORE_SLIDER)
    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(
            toggle_group,
            inputs=processors_checkbox_group,
            outputs=[FACE_LANDMARKER_GROUP]
        )


def toggle_group(processors: List[str]) -> gradio.update:
    all_processors = get_processors_modules()
    all_face_processor_names = [processor.display_name for processor in all_processors if processor.is_face_processor]
    # Make the group visible if any face processor is selected
    for processor in processors:
        if processor in all_face_processor_names:
            return gradio.update(visible=True)
    return gradio.update(visible=False)


def update_face_landmarker_model(face_landmarker_model: FaceLandmarkerModel) -> gradio.update:
    face_landmarker = FaceLandmarker()
    face_landmarker.clear_inference_pool()
    state_manager.set_item('face_landmarker_model', face_landmarker_model)

    if face_landmarker.pre_check():
        gradio.update(value=state_manager.get_item('face_landmarker_model'))
    return gradio.update()


def update_face_landmarker_score(face_landmarker_score: Score) -> None:
    state_manager.set_item('face_landmarker_score', face_landmarker_score)
