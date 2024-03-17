import gradio
from typing import Optional

import facefusion.choices
import facefusion.globals
from facefusion import wording

EXECUTION_QUEUE_COUNT_SLIDER: Optional[gradio.Slider] = None


def render() -> None:
    global EXECUTION_QUEUE_COUNT_SLIDER

    EXECUTION_QUEUE_COUNT_SLIDER = gradio.Slider(
        label=wording.get('uis.execution_queue_count_slider'),
        value=facefusion.globals.execution_queue_count,
        visible=False,
        step=facefusion.choices.execution_queue_count_range[1] - facefusion.choices.execution_queue_count_range[0],
        minimum=facefusion.choices.execution_queue_count_range[0],
        maximum=facefusion.choices.execution_queue_count_range[-1]
    )


def listen() -> None:
    EXECUTION_QUEUE_COUNT_SLIDER.change(update_execution_queue_count, inputs=EXECUTION_QUEUE_COUNT_SLIDER)


def update_execution_queue_count(execution_queue_count: int = 1) -> None:
    facefusion.globals.execution_queue_count = execution_queue_count
