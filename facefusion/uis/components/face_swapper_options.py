from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager, wording
from facefusion.common_helper import get_first
from facefusion.processors import choices as processors_choices
from facefusion.processors.classes.face_swapper import FaceSwapper
from facefusion.processors.core import load_processor_module
from facefusion.processors.typing import FaceSwapperModel
from facefusion.uis.core import get_ui_component, register_ui_component

FACE_SWAPPER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SWAPPER_PIXEL_BOOST_DROPDOWN: Optional[gradio.Dropdown] = None
PROCESSOR_KEY = 'Face Swapper'


def render() -> None:
    global FACE_SWAPPER_MODEL_DROPDOWN
    global FACE_SWAPPER_PIXEL_BOOST_DROPDOWN

    FACE_SWAPPER_MODEL_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.face_swapper_model_dropdown'),
        choices=FaceSwapper().list_models(),
        value=state_manager.get_item('face_swapper_model'),
        visible=PROCESSOR_KEY in state_manager.get_item('processors')
    )
    FACE_SWAPPER_PIXEL_BOOST_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.face_swapper_pixel_boost_dropdown'),
        choices=processors_choices.face_swapper_set.get(state_manager.get_item('face_swapper_model')),
        value=state_manager.get_item('face_swapper_pixel_boost'),
        visible=PROCESSOR_KEY in state_manager.get_item('processors')
    )
    register_ui_component('face_swapper_model_dropdown', FACE_SWAPPER_MODEL_DROPDOWN)
    register_ui_component('face_swapper_pixel_boost_dropdown', FACE_SWAPPER_PIXEL_BOOST_DROPDOWN)


def listen() -> None:
    FACE_SWAPPER_MODEL_DROPDOWN.change(update_face_swapper_model, inputs=FACE_SWAPPER_MODEL_DROPDOWN,
                                       outputs=[FACE_SWAPPER_MODEL_DROPDOWN, FACE_SWAPPER_PIXEL_BOOST_DROPDOWN])
    FACE_SWAPPER_PIXEL_BOOST_DROPDOWN.change(update_face_swapper_pixel_boost, inputs=FACE_SWAPPER_PIXEL_BOOST_DROPDOWN)

    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(remote_update, inputs=processors_checkbox_group,
                                         outputs=[FACE_SWAPPER_MODEL_DROPDOWN, FACE_SWAPPER_PIXEL_BOOST_DROPDOWN])


def remote_update(processors: List[str]) -> Tuple[gradio.update, gradio.update]:
    has_face_swapper = 'Face Swapper' in processors
    return gradio.update(visible=has_face_swapper), gradio.update(visible=has_face_swapper)


def update_face_swapper_model(face_swapper_model: FaceSwapperModel) -> Tuple[gradio.update, gradio.update]:
    face_swapper_module = load_processor_module(PROCESSOR_KEY)
    face_swapper_module.clear_inference_pool()
    state_manager.set_item('face_swapper_model', face_swapper_model)

    if face_swapper_module.pre_check():
        face_swapper_pixel_boost_choices = processors_choices.face_swapper_set.get(
            state_manager.get_item('face_swapper_model'))
        state_manager.set_item('face_swapper_pixel_boost', get_first(face_swapper_pixel_boost_choices))
        return gradio.update(value=state_manager.get_item('face_swapper_model')), gradio.update(
            value=state_manager.get_item('face_swapper_pixel_boost'), choices=face_swapper_pixel_boost_choices)
    return gradio.update(), gradio.update()


def update_face_swapper_pixel_boost(face_swapper_pixel_boost: str) -> None:
    state_manager.set_item('face_swapper_pixel_boost', face_swapper_pixel_boost)
