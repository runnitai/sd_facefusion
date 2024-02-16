import os
import time
from typing import Tuple, Optional

import gradio

import facefusion.globals
from facefusion import wording
from facefusion.core import limit_system_memory, conditional_process
from facefusion.filesystem import is_image, is_video, clear_temp
from facefusion.normalizer import normalize_output_path
from facefusion.uis.components import job_queue
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.ff_status import FFStatus
from facefusion.job_params import JobParams
from modules.paths_internal import script_path
from modules.sd_models import unload_model_weights, reload_model_weights
from modules.shared import sd_model

START_BUTTON: Optional[gradio.Button] = None
END_BUTTON: Optional[gradio.Button] = None
OUTPUT_FILES: Optional[gradio.Files] = None
OUTPUT_IMAGE: Optional[gradio.Image] = None
OUTPUT_VIDEO: Optional[gradio.Video] = None
OUTPUT_START: Optional[gradio.Button] = None
OUTPUT_CANCEL: Optional[gradio.Button] = None
PROGRESS_CHECKS = 0

OUTPUTS = []

def render() -> None:
    global OUTPUT_FILES
    global OUTPUT_IMAGE
    global OUTPUT_VIDEO
    global OUTPUT_START
    global OUTPUT_CANCEL

    with gradio.Row():
        OUTPUT_START = gradio.Button(
            value=wording.get('uis.start_button'),
            variant='primary',
            size='sm',
            elem_id="ff_start"
        )
        OUTPUT_CANCEL = gradio.Button(
            value="Cancel",
            size='sm',
            elem_id="ff_clear"
        )

    OUTPUT_FILES = gradio.Files(
        label="Outputs",
        visible=False,
        elem_class="output_element"
    )
    OUTPUT_IMAGE = gradio.Image(
        label=wording.get('uis.output_image_or_video'),
        visible=False,
        elem_class="output_element"
    )
    OUTPUT_VIDEO = gradio.Video(
        label=wording.get('uis.output_image_or_video'),
        visible=False,
        elem_class="output_element"
    )
    # A listener for this is set in the job_queue component
    register_ui_component('clear_button', OUTPUT_CANCEL)
    register_ui_component('output_files', OUTPUT_FILES)
    register_ui_component('output_image', OUTPUT_IMAGE)
    register_ui_component('output_video', OUTPUT_VIDEO)


def listen() -> None:
    output_path_textbox = get_ui_component('output_path_textbox')
    source_file = get_ui_component('source_file')
    # source_speaker = get_ui_component('source_speaker')
    target_file = get_ui_component('target_file')
    job_queue_table = get_ui_component('job_queue_table')
    ctl_elements = [OUTPUT_FILES, OUTPUT_IMAGE, OUTPUT_VIDEO, source_file, target_file, job_queue_table]
    if output_path_textbox:
        OUTPUT_START.click(start, _js="start_status", inputs=[output_path_textbox], outputs=ctl_elements,
                           show_progress=False)
    OUTPUT_CANCEL.click(clear, _js="stop_status", outputs=ctl_elements, show_progress=False)


def format_status():
    # Get status and progress
    status = FFStatus()
    if status.started and status.job_total > 0:
        progress = status.job_current / status.job_total
        progress = min(progress, 1)
    else:
        progress = 0

    # Calculate remaining time

    # Determine visibility and content of the output
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


def process_outputs() -> Tuple[gradio.update, gradio.update, gradio.update]:
    out_files = gradio.update(value=None, visible=False)
    out_image = gradio.update(value=None, visible=False)
    out_video = gradio.update(value=None, visible=False)
    if len(OUTPUTS) > 1:
        out_files = gradio.update(value=OUTPUTS, visible=True)
        out_image = gradio.update(value=None, visible=False)
        out_video = gradio.update(value=None, visible=False)
    elif len(OUTPUTS) == 1:
        out_files = gradio.update(value=None, visible=False)
        if is_image(OUTPUTS[0]):
            out_image = gradio.update(value=OUTPUTS[0], visible=True)
            out_video = gradio.update(value=None, visible=False)
        elif is_video(OUTPUTS[0]):
            out_image = gradio.update(value=None, visible=False)
            out_video = gradio.update(value=OUTPUTS[0], visible=True)

    return out_files, out_image, out_video


def start(output_path: str) -> Tuple[gradio.update, gradio.update, gradio.update, gradio.update, gradio.update]:
    """Start the FaceFusion process"""
    global OUTPUTS
    out_files = gradio.update(value=None, visible=False)
    out_image = gradio.update(value=None, visible=False)
    out_video = gradio.update(value=None, visible=False)
    src_files = gradio.update()
    tgt_file = gradio.update(visible=True, value=None)
    queue_table = gradio.update(visible=True, value=None)
    queue = job_queue.JOB_QUEUE
    completed_jobs = job_queue.COMPLETED_JOBS
    status = FFStatus(True)
    shared_model = sd_model
    if shared_model is not None:
        unload_model_weights()
    OUTPUTS = []
    total_jobs = len(queue)
    if total_jobs > 0:
        status.start(queue, "Starting FaceFusion")
        print("Starting jobs from queue")
        job_idx = 1
        for job in queue:
            if status.cancelled:
                print("Job cancelled")
                reload_model_weights(sd_model)
                return out_files, out_image, out_video, src_files, tgt_file

            status.next(job, f"Starting job {job_idx} of {total_jobs}", job_idx == 1)
            for key in job.__dict__:
                if not key.startswith("__") and key in facefusion.globals.__dict__:
                    facefusion.globals.__dict__[key] = job.__dict__[key]
            print(f"Starting job {job_idx} of {total_jobs}: {job}")
            out_path = start_job(job)
            OUTPUTS.append(out_path)
            completed_jobs.append(job)
            job_idx += 1
        job_queue.clear()
    else:
        print("No jobs in queue, you should fix that.")
    try:
        if shared_model is not None:
            reload_model_weights(shared_model)
    except:
        pass
    job_queue.clear()
    out_files, out_image, out_video = process_outputs()
    clear_temp()
    current_target_path = facefusion.globals.target_path
    if is_video(current_target_path):
        current_target_ext = os.path.splitext(current_target_path)[1]
        video_wav_path = current_target_path.replace(current_target_ext, ".wav")
        if video_wav_path in facefusion.globals.source_paths:
            facefusion.globals.source_paths.remove(video_wav_path)
            src_files = gradio.update(value=facefusion.globals.source_paths)
    facefusion.globals.target_path = ""
    facefusion.globals.reference_face_dict = {}
    facefusion.globals.mask_enabled_times = []
    facefusion.globals.mask_disabled_times = [0]
    facefusion.globals.trim_frame_start = None
    facefusion.globals.trim_frame_end = None
    facefusion.globals.output_path = os.path.join(script_path, "outputs", "facefusion")
    status.finish(f"Successfully processed {total_jobs} jobs.")
    return out_files, out_image, out_video, src_files, tgt_file, queue_table


def start_job(job: JobParams):
    out_path = os.path.join(script_path, "outputs", "facefusion")
    job.output_path = normalize_output_path(job.source_paths, job.target_path, out_path)
    limit_system_memory()
    conditional_process(job)
    output_path = job.output_path
    return output_path


def clear() -> Tuple[gradio.update, gradio.update, gradio.update, gradio.update, gradio.update]:
    status = FFStatus()
    status.cancel()
    src_files = gradio.update(value=None)
    tgt_file = gradio.update(visible=True, value=None)
    queue_table = gradio.update(visible=True, value=None)
    if facefusion.globals.target_path:
        clear_temp()
    out_files, out_image, out_video = process_outputs()
    return out_files, out_image, out_video, src_files, tgt_file, queue_table


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
