from typing import List, Optional, Union

import gradio

from facefusion import wording, state_manager
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.uis.typing import File

STYLE_TARGET_IMAGE: Optional[gradio.Image] = None


def render() -> None:
    global STYLE_TARGET_IMAGE
    STYLE_TARGET_IMAGE = gradio.Image(
        label=wording.get('uis.style_target_image'),
        value=state_manager.get_item('style_transfer_image'),
        visible='style_transfer' in state_manager.get_item('processors'),
        type='filepath',
    )

    register_ui_component('style_target_image', STYLE_TARGET_IMAGE)


def listen() -> None:
    STYLE_TARGET_IMAGE.change(update_style_target, inputs=STYLE_TARGET_IMAGE)
    STYLE_TARGET_IMAGE.clear(update_style_target, inputs=STYLE_TARGET_IMAGE)

    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(remote_update, inputs=processors_checkbox_group, outputs=STYLE_TARGET_IMAGE)


def remote_update(processors: List[str]) -> gradio.update:
    has_style_transfer = 'style_transfer' in processors
    return gradio.update(visible=has_style_transfer)


def update_style_target(style_target: Union[None, str]):
    print('style_target:', style_target)
    state_manager.set_item('style_transfer_image', style_target)
