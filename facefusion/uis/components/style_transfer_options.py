from typing import List, Optional

import gradio

from facefusion import wording, state_manager
from facefusion.filesystem import has_image
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.uis.typing import File

STYLE_TARGET_IMAGE: Optional[gradio.Image] = None
STYLE_TARGET_GALLERY: Optional[gradio.Gallery] = None
PROCESSOR_KEY = 'Style Transfer'


def render() -> None:
    global STYLE_TARGET_IMAGE, STYLE_TARGET_GALLERY
    style_images = state_manager.get_item('style_transfer_images')
    STYLE_TARGET_IMAGE = gradio.File(
        label=wording.get('uis.style_target_image'),
        file_types=['image'],
        value=state_manager.get_item('style_transfer_images'),
        file_count='multiple',
        visible=PROCESSOR_KEY in state_manager.get_item('processors') and not has_image(style_images)
    )
    STYLE_TARGET_GALLERY = gradio.Gallery(
        label=wording.get('uis.style_target_image'),
        value=state_manager.get_item('style_transfer_images'),
        visible=PROCESSOR_KEY in state_manager.get_item('processors') and has_image(style_images)
    )
    register_ui_component('style_target_images', STYLE_TARGET_IMAGE)
    register_ui_component('style_target_gallery', STYLE_TARGET_GALLERY)


def listen() -> None:
    STYLE_TARGET_IMAGE.change(update_style_target, inputs=STYLE_TARGET_IMAGE, outputs=STYLE_TARGET_GALLERY)
    STYLE_TARGET_IMAGE.clear(update_style_target, inputs=STYLE_TARGET_IMAGE, outputs=STYLE_TARGET_GALLERY)

    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(remote_update, inputs=processors_checkbox_group, outputs=STYLE_TARGET_IMAGE)


def remote_update(processors: List[str]) -> gradio.update:
    has_style_transfer = 'Style Transfer' in processors
    return gradio.update(visible=has_style_transfer)


def update_style_target(style_target: List[File]):
    print('style_target:', style_target)
    target_names = [style_target.name for style_target in style_target]
    state_manager.set_item('style_transfer_images', target_names)
    return gradio.update(value=target_names, visible=has_image(target_names))


