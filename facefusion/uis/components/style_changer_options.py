from typing import List, Optional, Tuple

import gradio

from facefusion import wording, state_manager
from facefusion.processors.classes.style_changer import StyleChanger
from facefusion.uis.core import get_ui_component, register_ui_component

STYLE_CHANGER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
STYLE_TARGET_RADIO: Optional[gradio.Radio] = None
STYLE_CHANGER_SKIP_HEAD_CHECKBOX: Optional[gradio.Checkbox] = None
PROCESSOR_KEY = 'Style Changer'


def render() -> None:
    global STYLE_CHANGER_MODEL_DROPDOWN
    global STYLE_TARGET_RADIO
    global STYLE_CHANGER_SKIP_HEAD_CHECKBOX
    STYLE_CHANGER_MODEL_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.style_changer_model_dropdown'),
        choices=StyleChanger().list_models(),
        value=state_manager.get_item('style_changer_model'),
        visible=PROCESSOR_KEY in state_manager.get_item('processors'),
        elem_id='style_changer_model_dropdown'
    )
    STYLE_TARGET_RADIO = gradio.Radio(
        label=wording.get('uis.style_changer_target_radio'),
        choices=["source", "target", "source head/target bg"],
        value=str(state_manager.get_item('style_changer_target')),
        visible=PROCESSOR_KEY in state_manager.get_item('processors'),
        elem_id='style_target_radio'
    )
    STYLE_CHANGER_SKIP_HEAD_CHECKBOX = gradio.Checkbox(
        label=wording.get('uis.style_changer_skip_head_checkbox'),
        value=state_manager.get_item('style_changer_skip_head'),
        visible=PROCESSOR_KEY in state_manager.get_item('processors'),
        elem_id='style_changer_skip_head_checkbox'
    )

    register_ui_component('style_changer_model_dropdown', STYLE_CHANGER_MODEL_DROPDOWN)
    register_ui_component('style_target_radio', STYLE_TARGET_RADIO)
    register_ui_component('style_changer_skip_head_checkbox', STYLE_CHANGER_SKIP_HEAD_CHECKBOX)


def listen() -> None:
    STYLE_CHANGER_MODEL_DROPDOWN.change(update_style_changer_model, inputs=STYLE_CHANGER_MODEL_DROPDOWN)
    STYLE_TARGET_RADIO.change(update_style_target, inputs=STYLE_TARGET_RADIO, outputs=[STYLE_CHANGER_SKIP_HEAD_CHECKBOX])
    STYLE_CHANGER_SKIP_HEAD_CHECKBOX.change(update_style_changer_skip_head, inputs=STYLE_CHANGER_SKIP_HEAD_CHECKBOX)

    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(remote_update, inputs=processors_checkbox_group,
                                         outputs=[STYLE_CHANGER_MODEL_DROPDOWN, STYLE_TARGET_RADIO,
                                                  STYLE_CHANGER_SKIP_HEAD_CHECKBOX])


def remote_update(processors: List[str]) -> Tuple[gradio.update, gradio.update, gradio.update]:
    has_style_changer = 'Style Changer' in processors
    return gradio.update(visible=has_style_changer), gradio.update(visible=has_style_changer), gradio.update(
        visible=has_style_changer)


def update_style_changer_skip_head(style_changer_skip_head: bool):
    state_manager.set_item('style_changer_skip_head', style_changer_skip_head)
    return


def update_style_changer_model(style_changer_model: str):
    StyleChanger().clear_inference_pool()
    state_manager.set_item('style_changer_model', style_changer_model)


def update_style_target(style_target: str) -> gradio.update:
    state_manager.set_item('style_changer_target', style_target)
    return gradio.update(visible=style_target != 'source head/target bg')
