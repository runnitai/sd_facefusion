from typing import List, Optional

import gradio

from facefusion import state_manager, wording
from facefusion.execution import get_execution_provider_choices
from facefusion.processors.core import clear_processors_modules
from facefusion.typing import ExecutionProviderKey
from facefusion.workers.core import clear_worker_modules

EXECUTION_PROVIDERS_CHECKBOX_GROUP: Optional[gradio.CheckboxGroup] = None


def render() -> None:
    global EXECUTION_PROVIDERS_CHECKBOX_GROUP

    EXECUTION_PROVIDERS_CHECKBOX_GROUP = gradio.CheckboxGroup(
        label=wording.get('uis.execution_providers_checkbox_group'),
        choices=get_execution_provider_choices(),
        value=state_manager.get_item('execution_providers')
    )


def listen() -> None:
    EXECUTION_PROVIDERS_CHECKBOX_GROUP.change(update_execution_providers, inputs=EXECUTION_PROVIDERS_CHECKBOX_GROUP,
                                              outputs=EXECUTION_PROVIDERS_CHECKBOX_GROUP)


def update_execution_providers(execution_providers: List[ExecutionProviderKey]) -> gradio.update:
    clear_worker_modules()
    clear_processors_modules(state_manager.get_item('processors'))
    execution_providers = execution_providers or get_execution_provider_choices()
    state_manager.set_item('execution_providers', execution_providers)
    return gradio.update(value=state_manager.get_item('execution_providers'))
