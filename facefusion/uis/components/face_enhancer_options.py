from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager, wording
from facefusion.common_helper import calc_int_step
from facefusion.processors import choices as processors_choices
from facefusion.processors.classes.face_enhancer import FaceEnhancer
from facefusion.processors.core import load_processor_module
from facefusion.processors.typing import FaceEnhancerModel
from facefusion.uis.core import get_ui_component, register_ui_component

FACE_ENHANCER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_ENHANCER_BLEND_SLIDER: Optional[gradio.Slider] = None
FACE_ENHANCER_SMART_ENHANCE_CHECKBOX: Optional[gradio.Checkbox] = None
FACE_ENHANCER_SMART_MINIMUM_SIZE_SLIDER: Optional[gradio.Slider] = None
PROCESSOR_KEY = 'Face Enhancer'


def render() -> None:
    global FACE_ENHANCER_MODEL_DROPDOWN
    global FACE_ENHANCER_BLEND_SLIDER
    global FACE_ENHANCER_SMART_ENHANCE_CHECKBOX
    global FACE_ENHANCER_SMART_MINIMUM_SIZE_SLIDER

    # Initialize smart enhance state items if not already set
    if state_manager.get_item('face_enhancer_smart_enhance') is None:
        state_manager.set_item('face_enhancer_smart_enhance', True)
    if state_manager.get_item('face_enhancer_smart_minimum_size') is None:
        state_manager.set_item('face_enhancer_smart_minimum_size', 250)

    FACE_ENHANCER_MODEL_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.face_enhancer_model_dropdown'),
        choices=FaceEnhancer().list_models(),
        value=state_manager.get_item('face_enhancer_model'),
        visible=PROCESSOR_KEY in state_manager.get_item('processors')
    )
    FACE_ENHANCER_BLEND_SLIDER = gradio.Slider(
        label=wording.get('uis.face_enhancer_blend_slider'),
        value=state_manager.get_item('face_enhancer_blend'),
        step=calc_int_step(processors_choices.face_enhancer_blend_range),
        minimum=processors_choices.face_enhancer_blend_range[0],
        maximum=processors_choices.face_enhancer_blend_range[-1],
        visible=PROCESSOR_KEY in state_manager.get_item('processors')
    )
    FACE_ENHANCER_SMART_ENHANCE_CHECKBOX = gradio.Checkbox(
        label=wording.get('uis.face_enhancer_smart_enhance_checkbox'),
        value=state_manager.get_item('face_enhancer_smart_enhance') if state_manager.get_item('face_enhancer_smart_enhance') is not None else True,
        visible=PROCESSOR_KEY in state_manager.get_item('processors')
    )
    FACE_ENHANCER_SMART_MINIMUM_SIZE_SLIDER = gradio.Slider(
        label=wording.get('uis.face_enhancer_smart_minimum_size_slider'),
        value=state_manager.get_item('face_enhancer_smart_minimum_size') if state_manager.get_item('face_enhancer_smart_minimum_size') is not None else 250,
        step=calc_int_step(processors_choices.face_enhancer_smart_minimum_size_range),
        minimum=processors_choices.face_enhancer_smart_minimum_size_range[0],
        maximum=processors_choices.face_enhancer_smart_minimum_size_range[-1],
        visible=PROCESSOR_KEY in state_manager.get_item('processors') and (state_manager.get_item('face_enhancer_smart_enhance') if state_manager.get_item('face_enhancer_smart_enhance') is not None else True)
    )
    register_ui_component('face_enhancer_model_dropdown', FACE_ENHANCER_MODEL_DROPDOWN)
    register_ui_component('face_enhancer_blend_slider', FACE_ENHANCER_BLEND_SLIDER)
    register_ui_component('face_enhancer_smart_enhance_checkbox', FACE_ENHANCER_SMART_ENHANCE_CHECKBOX)
    register_ui_component('face_enhancer_smart_minimum_size_slider', FACE_ENHANCER_SMART_MINIMUM_SIZE_SLIDER)


def listen() -> None:
    FACE_ENHANCER_MODEL_DROPDOWN.change(update_face_enhancer_model, inputs=FACE_ENHANCER_MODEL_DROPDOWN,
                                        outputs=FACE_ENHANCER_MODEL_DROPDOWN)
    FACE_ENHANCER_BLEND_SLIDER.release(update_face_enhancer_blend, inputs=FACE_ENHANCER_BLEND_SLIDER)
    FACE_ENHANCER_SMART_ENHANCE_CHECKBOX.change(update_face_enhancer_smart_enhance, 
                                               inputs=FACE_ENHANCER_SMART_ENHANCE_CHECKBOX,
                                               outputs=FACE_ENHANCER_SMART_MINIMUM_SIZE_SLIDER)
    FACE_ENHANCER_SMART_MINIMUM_SIZE_SLIDER.release(update_face_enhancer_smart_minimum_size, 
                                                   inputs=FACE_ENHANCER_SMART_MINIMUM_SIZE_SLIDER)

    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(remote_update, inputs=processors_checkbox_group,
                                         outputs=[FACE_ENHANCER_MODEL_DROPDOWN, FACE_ENHANCER_BLEND_SLIDER,
                                                 FACE_ENHANCER_SMART_ENHANCE_CHECKBOX, FACE_ENHANCER_SMART_MINIMUM_SIZE_SLIDER])


def remote_update(processors: List[str]) -> Tuple[gradio.update, gradio.update, gradio.update, gradio.update]:
    has_face_enhancer = 'Face Enhancer' in processors
    smart_enhance_enabled = has_face_enhancer and state_manager.get_item('face_enhancer_smart_enhance')
    return (gradio.update(visible=has_face_enhancer), 
            gradio.update(visible=has_face_enhancer),  
            gradio.update(visible=has_face_enhancer),
            gradio.update(visible=smart_enhance_enabled))


def update_face_enhancer_model(face_enhancer_model: FaceEnhancerModel) -> gradio.update:
    face_enhancer_module = load_processor_module(PROCESSOR_KEY)
    face_enhancer_module.clear_inference_pool()
    state_manager.set_item('face_enhancer_model', face_enhancer_model)

    if face_enhancer_module.pre_check():
        return gradio.update(value=state_manager.get_item('face_enhancer_model'))
    return gradio.update()


def update_face_enhancer_blend(face_enhancer_blend: float) -> None:
    state_manager.set_item('face_enhancer_blend', int(face_enhancer_blend))


def update_face_enhancer_smart_enhance(face_enhancer_smart_enhance: bool) -> gradio.update:
    state_manager.set_item('face_enhancer_smart_enhance', face_enhancer_smart_enhance)
    return gradio.update(visible=face_enhancer_smart_enhance)


def update_face_enhancer_smart_minimum_size(face_enhancer_smart_minimum_size: float) -> None:
    state_manager.set_item('face_enhancer_smart_minimum_size', int(face_enhancer_smart_minimum_size))
