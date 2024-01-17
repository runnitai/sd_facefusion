from typing import Optional, List

import gradio

from facefusion.uis import choices
from facefusion.uis.core import register_ui_component

JOB_QUEUE_OPTIONS_CHECKBOX_GROUP: Optional[gradio.Checkboxgroup] = None
CLEAR_SOURCE = False
CLEAR_TARGET = False


def render() -> None:
    global JOB_QUEUE_OPTIONS_CHECKBOX_GROUP

    value = ['Clear Target']
    JOB_QUEUE_OPTIONS_CHECKBOX_GROUP = gradio.Checkboxgroup(
        label="Job Queue Options",
        choices=choices.job_queue_options,
        value=value,
        elem_id='ff_job_queue_options_checkbox_group',
        visible=False
    )
    register_ui_component('ff_job_queue_options_checkbox_group', JOB_QUEUE_OPTIONS_CHECKBOX_GROUP)


def listen() -> None:
    JOB_QUEUE_OPTIONS_CHECKBOX_GROUP.change(update, inputs=JOB_QUEUE_OPTIONS_CHECKBOX_GROUP)


def update(queue_options: List[str]) -> None:
    global CLEAR_SOURCE
    global CLEAR_TARGET
    CLEAR_SOURCE = 'Clear Source' in queue_options
    CLEAR_TARGET = 'Clear Target' in queue_options
