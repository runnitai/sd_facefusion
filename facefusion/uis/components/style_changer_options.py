from typing import List, Optional, Tuple

import gradio

from facefusion import wording, state_manager
from facefusion.processors import choices as frame_processors_choices
from facefusion.processors.modules.style_changer import clear_inference_pool
from facefusion.uis.components.source import check_swap_source_style
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.uis.typing import File

STYLE_CHANGER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
STYLE_TARGET_RADIO: Optional[gradio.Radio] = None
STYLE_CHANGER_SKIP_HEAD_CHECKBOX: Optional[gradio.Checkbox] = None


def render() -> None:
    global STYLE_CHANGER_MODEL_DROPDOWN
    global STYLE_TARGET_RADIO
    global STYLE_CHANGER_SKIP_HEAD_CHECKBOX
    STYLE_CHANGER_MODEL_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.style_changer_model_dropdown'),
        choices=frame_processors_choices.style_changer_models,
        value=state_manager.get_item('style_changer_model'),
        visible='style_changer' in state_manager.get_item('processors'),
        elem_id='style_changer_model_dropdown'
    )
    STYLE_TARGET_RADIO = gradio.Radio(
        label=wording.get('uis.style_changer_target_radio'),
        choices=["source", "target"],
        value=str(state_manager.get_item('style_changer_target')),
        visible='style_changer' in state_manager.get_item('processors'),
        elem_id='style_target_radio'
    )
    STYLE_CHANGER_SKIP_HEAD_CHECKBOX = gradio.Checkbox(
        label=wording.get('uis.style_changer_skip_head_checkbox'),
        value=state_manager.get_item('style_changer_skip_head'),
        visible='style_changer' in state_manager.get_item('processors'),
        elem_id='style_changer_skip_head_checkbox'
    )

    register_ui_component('style_changer_model_dropdown', STYLE_CHANGER_MODEL_DROPDOWN)
    register_ui_component('style_target_radio', STYLE_TARGET_RADIO)
    register_ui_component('style_changer_skip_head_checkbox', STYLE_CHANGER_SKIP_HEAD_CHECKBOX)


def listen() -> None:
    STYLE_CHANGER_MODEL_DROPDOWN.change(update_style_changer_model, inputs=STYLE_CHANGER_MODEL_DROPDOWN)
    source_file = get_ui_component('source_file')
    source_file_2 = get_ui_component('source_file_2')
    STYLE_TARGET_RADIO.change(update_style_target, inputs=[STYLE_TARGET_RADIO, source_file, source_file_2],
                              outputs=[source_file, source_file_2])
    STYLE_CHANGER_SKIP_HEAD_CHECKBOX.change(update_style_changer_skip_head, inputs=STYLE_CHANGER_SKIP_HEAD_CHECKBOX)

    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(remote_update, inputs=processors_checkbox_group,
                                         outputs=[STYLE_CHANGER_MODEL_DROPDOWN, STYLE_TARGET_RADIO,
                                                  STYLE_CHANGER_SKIP_HEAD_CHECKBOX])


def remote_update(processors: List[str]) -> Tuple[gradio.update, gradio.update, gradio.update]:
    has_style_changer = 'style_changer' in processors
    return gradio.update(visible=has_style_changer), gradio.update(visible=has_style_changer), gradio.update(
        visible=has_style_changer)


def update_style_changer_skip_head(style_changer_skip_head: bool):
    state_manager.set_item('style_changer_skip_head', style_changer_skip_head)
    return


def update_style_changer_model(style_changer_model: str):
    clear_inference_pool()
    state_manager.set_item('style_changer_model', style_changer_model)
    return


def update_style_target(style_target: str, source_file: List[File], source_file_2: List[File]) -> Tuple[
    gradio.update, gradio.update]:
    state_manager.set_item('style_changer_target', style_target)
    return check_swap_source_style(source_file), check_swap_source_style(source_file_2)
