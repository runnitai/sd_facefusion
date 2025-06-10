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


def render() -> None:
    global OUTPUT_IMAGE
    global OUTPUT_VIDEO
    global OUTPUT_STATUS

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


def listen() -> None:
    register_ui_component('output_image', OUTPUT_IMAGE)
    register_ui_component('output_video', OUTPUT_VIDEO)
    register_ui_component('output_status', OUTPUT_STATUS)


def update_status():
    """Update status display - now simplified since we use Gradio's built-in progress"""
    status = FFStatus()
    out_video = gradio.update()
    out_image = gradio.update()
    if status.preview_image and os.path.exists(status.preview_image):
        preview_image = gradio.update(value=status.preview_image, visible=True)
    else:
        preview_image = gradio.update(visible=False)
    return gradio.update(visible=True, value=format_status()), out_image, preview_image, out_video


def format_status():
    """Format status display - simplified since progress is now handled by Gradio"""
    status = FFStatus()
    
    status_string = status.status if status.status is not None else ""
    
    # Simple status display without the hacky progress bar
    if status.started:
        return f"<div class='status-active'>{status_string}</div>"
    else:
        return f"<div class='status-idle'>{status_string}</div>"


def calc_time_left(progress, threshold, label, force_display):
    """Legacy function - kept for compatibility but simplified"""
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
