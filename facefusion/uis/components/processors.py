from typing import List, Optional

import gradio

from facefusion import state_manager, wording
from facefusion.filesystem import list_directory
from facefusion.processors.core import clear_processors_modules, get_processors_modules, list_processors
from facefusion.uis.core import register_ui_component

PROCESSORS_CHECKBOX_GROUP: Optional[gradio.CheckboxGroup] = None


def render() -> None:
    global PROCESSORS_CHECKBOX_GROUP

    PROCESSORS_CHECKBOX_GROUP = gradio.CheckboxGroup(
        label=wording.get('uis.processors_checkbox_group'),
        choices=list_processors(),
        value=state_manager.get_item('processors')
    )
    register_ui_component('processors_checkbox_group', PROCESSORS_CHECKBOX_GROUP)


def listen() -> None:
    PROCESSORS_CHECKBOX_GROUP.change(update_processors, inputs=PROCESSORS_CHECKBOX_GROUP,
                                     outputs=PROCESSORS_CHECKBOX_GROUP)


def update_processors(processors: List[str]) -> gradio.update:
    """Update processors and maintain order/state properly"""
    
    # Get current processors before clearing
    current_processors = state_manager.get_item('processors') or []
    
    # Add debug logging
    from facefusion import logger
    logger.info(f"Updating processors from {current_processors} to {processors}", __name__)
    
    # Only clear processors that are being removed
    processors_to_clear = [p for p in current_processors if p not in processors]
    if processors_to_clear:
        logger.info(f"Clearing processors: {processors_to_clear}", __name__)
        clear_processors_modules(processors_to_clear)
    
    # Update the processors list
    state_manager.set_item('processors', processors)

    # Pre-check only new processors to avoid unnecessary reloading
    new_processors = [p for p in processors if p not in current_processors]
    if new_processors:
        logger.info(f"Pre-checking new processors: {new_processors}", __name__)
        for processor_module in get_processors_modules(new_processors):
            if not processor_module.pre_check():
                logger.warn(f"Pre-check failed for processor: {processor_module.display_name}", __name__)
                return gradio.update()
    
    return gradio.update(value=state_manager.get_item('processors'),
                         choices=sort_processors(state_manager.get_item('processors')))


def sort_processors(processors: List[str]) -> List[str]:
    available_processors = list_processors()
    return sorted(available_processors,
                  key=lambda processor: processors.index(processor) if processor in processors else len(processors))
