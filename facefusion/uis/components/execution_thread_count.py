import gradio
from typing import Optional

import facefusion.choices
import facefusion.globals
from facefusion import wording

EXECUTION_THREAD_COUNT_SLIDER: Optional[gradio.Slider] = None


def render() -> None:
    global EXECUTION_THREAD_COUNT_SLIDER
    # Get the total VRAM from CUDA
    EXECUTION_THREAD_COUNT_SLIDER = gradio.Slider(
        label=wording.get('uis.execution_thread_count_slider'),
        value=facefusion.globals.execution_thread_count,
        visible=False,
        step=facefusion.choices.execution_thread_count_range[1] - facefusion.choices.execution_thread_count_range[0],
        minimum=facefusion.choices.execution_thread_count_range[0],
        maximum=facefusion.choices.execution_thread_count_range[-1]
    )


def listen() -> None:
    EXECUTION_THREAD_COUNT_SLIDER.change(update_execution_thread_count, inputs=EXECUTION_THREAD_COUNT_SLIDER)


def update_execution_thread_count(execution_thread_count: int = 1) -> None:
    facefusion.globals.execution_thread_count = execution_thread_count
