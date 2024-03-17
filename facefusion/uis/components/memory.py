import gradio
from typing import Optional

import facefusion.choices
import facefusion.globals
from facefusion import wording
from facefusion.typing import VideoMemoryStrategy

VIDEO_MEMORY_STRATEGY: Optional[gradio.Dropdown] = None
SYSTEM_MEMORY_LIMIT_SLIDER: Optional[gradio.Slider] = None


def render() -> None:
    global VIDEO_MEMORY_STRATEGY
    global SYSTEM_MEMORY_LIMIT_SLIDER

    VIDEO_MEMORY_STRATEGY = gradio.Dropdown(
        label=wording.get('uis.video_memory_strategy_dropdown'),
        choices=facefusion.choices.video_memory_strategies,
        value=facefusion.globals.video_memory_strategy
    )
    SYSTEM_MEMORY_LIMIT_SLIDER = gradio.Slider(
        label=wording.get('uis.system_memory_limit_slider'),
        step=facefusion.choices.system_memory_limit_range[1] - facefusion.choices.system_memory_limit_range[0],
        minimum=facefusion.choices.system_memory_limit_range[0],
        maximum=facefusion.choices.system_memory_limit_range[-1],
        value=facefusion.globals.system_memory_limit
    )


def listen() -> None:
    VIDEO_MEMORY_STRATEGY.change(update_video_memory_strategy, inputs=VIDEO_MEMORY_STRATEGY)
    SYSTEM_MEMORY_LIMIT_SLIDER.change(update_system_memory_limit, inputs=SYSTEM_MEMORY_LIMIT_SLIDER)


def update_video_memory_strategy(video_memory_strategy: VideoMemoryStrategy) -> None:
    facefusion.globals.video_memory_strategy = video_memory_strategy


def update_system_memory_limit(system_memory_limit: int) -> None:
    facefusion.globals.system_memory_limit = system_memory_limit
