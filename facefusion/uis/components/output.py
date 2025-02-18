import os
import time
from typing import Optional

import gradio

from facefusion import state_manager, wording
from facefusion.ff_status import FFStatus
from facefusion.filesystem import get_output_path_auto
from facefusion.uis.core import register_ui_component, get_ui_component

OUTPUT_IMAGE: Optional[gradio.Image] = None
OUTPUT_VIDEO: Optional[gradio.Video] = None
OUTPUT_STATUS: Optional[gradio.HTML] = None
CHECK_STATUS_BUTTON: Optional[gradio.Button] = None


def render() -> None:
    global OUTPUT_IMAGE
    global OUTPUT_VIDEO
    global OUTPUT_STATUS
    global CHECK_STATUS_BUTTON

    out_dir = get_output_path_auto()
    state_manager.init_item('output_path', out_dir)
    OUTPUT_STATUS = gradio.HTML(elem_id="ff_status", value=format_status(), visible=False)
    OUTPUT_IMAGE = gradio.Image(
        label=wording.get('uis.output_image_or_video'),
        visible=False
    )
    OUTPUT_VIDEO = gradio.Video(
        label=wording.get('uis.output_image_or_video')
    )
    CHECK_STATUS_BUTTON = gradio.Button(
        value=wording.get('uis.check_status_button'),
        variant='primary',
        size='sm',
        visible=False,
    )


def listen() -> None:
    register_ui_component('output_image', OUTPUT_IMAGE)
    register_ui_component('output_video', OUTPUT_VIDEO)
    register_ui_component('output_status', OUTPUT_STATUS)
    register_ui_component('check_status', CHECK_STATUS_BUTTON)

    output_image = get_ui_component('output_image')
    output_video = get_ui_component('output_video')
    preview_image = get_ui_component('preview_image')
    CHECK_STATUS_BUTTON.click(update_status, inputs=[],
                              outputs=[OUTPUT_STATUS, output_image, preview_image, output_video], show_progress=False)


def update_status():
    status = FFStatus()
    out_video = gradio.update()
    out_image = gradio.update()
    if status.preview_image and os.path.exists(status.preview_image):
        preview_image = gradio.update(value=status.preview_image, visible=True)
    else:
        preview_image = gradio.update(visible=False)
    return gradio.update(visible=True, value=format_status()), out_image, preview_image, out_video


def format_status():
    # Get status and progress
    status = FFStatus()
    if status.started and status.job_total > 0:
        progress = status.job_current / status.job_total
        progress = min(progress, 1)
    else:
        progress = 0

    status_string = status.status if status.status is not None else ""
    hidden_div = f"<div style='display:none' id='statusDiv' data-started={'true' if status.started else 'false'}></div>"
    status_string = f"{hidden_div}{status_string}"
    if 0 < progress < 1:
        time_left = calc_time_left(progress, threshold=60, label="ETA: ", force_display=True)
        progress_percentage = f"{time_left} - {int(progress * 100)}%"
        progress_bar = f"""
            <div class='progressDiv'>
                <div class='progress' style="overflow:visible;width:{progress * 100}%;white-space:nowrap;">
                    {"&nbsp;" * 2 + progress_percentage if progress > 0.01 else ""}
                </div>
            </div>
            """
        overlay = f"{status_string}{progress_bar}"
    else:
        overlay = status_string
    return overlay


def calc_time_left(progress, threshold, label, force_display):
    status = FFStatus()
    if progress == 0:
        return ""
    else:
        if status.time_start is None:
            time_since_start = 0
        else:
            time_since_start = time.time() - status.time_start
        eta = time_since_start / progress
        eta_relative = eta - time_since_start
        if (eta_relative > threshold and progress > 0.02) or force_display:
            if eta_relative > 86400:
                days = eta_relative // 86400
                remainder = days * 86400
                eta_relative -= remainder
                return f"{label}{days}:{time.strftime('%H:%M:%S', time.gmtime(eta_relative))}"
            if eta_relative > 3600:
                return label + time.strftime("%H:%M:%S", time.gmtime(eta_relative))
            elif eta_relative > 60:
                return label + time.strftime("%M:%S", time.gmtime(eta_relative))
            else:
                return label + time.strftime("%Ss", time.gmtime(eta_relative))
        else:
            return ""
