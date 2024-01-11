import logging
import os.path
from typing import Optional

import gradio

import facefusion.globals
from facefusion.job_params import JobParams
from facefusion.normalizer import normalize_output_path
from facefusion.uis.core import get_ui_component, register_ui_component

logger = logging.getLogger(__name__)

OUTPUT_ENQUEUE_BUTTON: Optional[gradio.Button] = None
CLEAR_QUEUE_BUTTON: Optional[gradio.Button] = None
REMOVE_LAST_BUTTON: Optional[gradio.Button] = None
TOGGLE_REMOVE_BUTTON: Optional[gradio.Button] = None
LAST_ELEMENT: Optional[gradio.Text] = None
JOB_QUEUE = []
COMPLETED_JOBS = []
JOB_QUEUE_TABLE: Optional[gradio.HTML] = None

CHECK_STATUS_BUTTON: Optional[gradio.Button] = None
STOP_CHECK_BUTTON: Optional[gradio.Button] = None
OUTPUT_STATUS: Optional[gradio.HTML] = None
LIVE_STATUS: Optional[gradio.HTML] = None


def queue_to_table():
    data = JOB_QUEUE
    if not data:
        return "<table id='queueTable'><tr><th>ID</th><th>Output Path</th><th>Target Path</th><th>Source Path</th></tr></table>"
    table_rows = ["<tr><th>ID</th><th>Output Path</th><th>Target Path</th><th>Source Path</th></tr>"]
    idx = 0
    for item in data:
        out_path = item.output_path
        if len(out_path) > 50:
            out_path = os.path.basename(out_path)
        src_path = item.source_paths
        if len(src_path) > 50:
            src_path = os.path.basename(src_path)
        tgt_path = item.target_path
        if len(tgt_path) > 50:
            tgt_path = os.path.basename(tgt_path)
        row = f"<tr class='selectRow' id='row{idx}' onclick='document.getElementById(\"ff_last_element\").value = {idx};'><td>{idx}</td><td>{out_path}</td><td>{tgt_path}</td><td>{src_path}</td></tr>"
        table_rows.append(row)
        idx += 1
    table_html = f"<table id='queueTable'>{''.join(table_rows)}</table>"
    return table_html


def render() -> None:
    global CHECK_STATUS_BUTTON
    global STOP_CHECK_BUTTON
    global LIVE_STATUS
    global OUTPUT_STATUS

    global JOB_QUEUE_TABLE
    global LAST_ELEMENT
    global OUTPUT_ENQUEUE_BUTTON
    global CLEAR_QUEUE_BUTTON
    global REMOVE_LAST_BUTTON
    global TOGGLE_REMOVE_BUTTON

    with gradio.Column(elem_classes=["queueRow"]):
        gradio.Markdown("##### Job Queue")
        JOB_QUEUE_TABLE = gradio.HTML(value=queue_to_table())
        register_ui_component('job_queue_table', JOB_QUEUE_TABLE)
        LAST_ELEMENT = gradio.Text(value="", label="Last Element", elem_id="ff_last_element", visible=False)
        with gradio.Row(elem_classes=["queueRow"]):
            with gradio.Column():
                CHECK_STATUS_BUTTON = gradio.Button(visible=False, elem_id="ff_check_status")
                from facefusion.uis.components.output import format_status
                OUTPUT_STATUS = gradio.HTML(elem_id="ff_status", value=format_status(), visible=False)

                OUTPUT_ENQUEUE_BUTTON = gradio.Button(
                    value="Add to Queue",
                    size='sm',
                    elem_id="ff_enqueue"
                )
            with gradio.Column():
                # TODO: Toggle the enabled state of this when clicking/unclicking a row
                REMOVE_LAST_BUTTON = gradio.Button(
                    value="Remove",
                    size='sm',
                    elem_id="ff_remove_last",
                    interactive=False,
                )
                TOGGLE_REMOVE_BUTTON = gradio.Button(
                    value="Enable Remove",
                    size='sm',
                    elem_id="ff_toggle_remove",
                    visible=False
                )

            with gradio.Column():
                CLEAR_QUEUE_BUTTON = gradio.Button(
                    value="Clear Queue",
                    size='sm',
                    elem_id="ff_clear_queue"
                )


def listen() -> None:
    source_image = get_ui_component('source_image')
    target_file = get_ui_component('target_file')
    output_files = get_ui_component('output_files')
    output_video = get_ui_component('output_video')
    preview_image = get_ui_component('preview_image')
    if source_image and target_file:
        OUTPUT_ENQUEUE_BUTTON.click(enqueue, inputs=[], outputs=[JOB_QUEUE_TABLE, source_image, target_file])
    CLEAR_QUEUE_BUTTON.click(clear, inputs=[], outputs=[JOB_QUEUE_TABLE, source_image, target_file])
    REMOVE_LAST_BUTTON.click(remove_last, _js="get_selected_row", inputs=[LAST_ELEMENT], outputs=[JOB_QUEUE_TABLE])
    TOGGLE_REMOVE_BUTTON.click(toggle_remove, inputs=[TOGGLE_REMOVE_BUTTON],
                               outputs=[REMOVE_LAST_BUTTON, TOGGLE_REMOVE_BUTTON])
    CHECK_STATUS_BUTTON.click(update_status, inputs=[],
                              outputs=[OUTPUT_STATUS, output_files, preview_image, output_video], show_progress=False)


def update_status():
    from facefusion.uis.components.output import format_status
    from facefusion.uis.components.output import STATUS
    out_video = gradio.update()
    out_files = gradio.update()
    if STATUS.preview_image:
        out_image = gradio.update(value=STATUS.preview_image, visible=True)
    else:
        out_image = gradio.update(visible=False)
    return gradio.update(visible=True, value=format_status()), out_files, out_image, out_video


def toggle_remove(toggle_button) -> gradio.update:
    global REMOVE_LAST_BUTTON
    global TOGGLE_REMOVE_BUTTON
    if toggle_button == "Enable Remove":
        return gradio.update(value="Remove", visible=True, interactive=True), gradio.update(value="Disable Remove",
                                                                                            visible=False)
    else:
        return gradio.update(value="Remove", visible=True, interactive=False), gradio.update(value="Enable Remove",
                                                                                             visible=False)


def clear() -> gradio.update:
    global JOB_QUEUE
    JOB_QUEUE = []
    temp_test_dir = os.path.join(os.path.dirname(facefusion.globals.output_path), "ff_debug")
    if os.path.exists(temp_test_dir):
        import shutil
        shutil.rmtree(temp_test_dir)
    queue_table = gradio.update(value=queue_to_table(), visible=True)
    target_file = gradio.update(value=None)
    source_image = gradio.update(value=None)
    return queue_table, target_file, source_image


def remove_last(last_value) -> gradio.update:
    global JOB_QUEUE
    # If the last value is a string, try and parse it as an int
    if isinstance(last_value, str):
        try:
            last_value = int(last_value)
        except ValueError:
            print(f"Could not parse {last_value} as int")
            last_value = -1
    # If the last value is an int, remove the corresponding element from the job queue
    if isinstance(last_value, int):
        if 0 < last_value <= len(JOB_QUEUE):
            print(f"Removing job {last_value} from queue")
            last_item = JOB_QUEUE.pop(last_value - 1)
            temp_test_dir = os.path.join(os.path.dirname(last_item.output_path), "ff_debug")
            temp_job_path = os.path.join(temp_test_dir, f"job_{last_item.id}.json")
            if os.path.exists(temp_job_path):
                os.remove(temp_job_path)
        else:
            print(f"Could not remove job {last_value} from queue")

    return gradio.update(value=queue_to_table(), visible=True)


def enqueue() -> gradio.update:
    global JOB_QUEUE
    # Enumerate all values in facefusion.globals to a dict
    global_dict = {}
    for key in facefusion.globals.__dict__:
        if not key.startswith("__"):
            global_dict[key] = facefusion.globals.__dict__[key]
    required_keys = ["output_path", "target_path", "source_paths"]
    # If any of the required keys are missing, don't add the job to the queue
    if any(key not in global_dict for key in required_keys):
        print(f"Missing required key in facefusion.globals")
        return gradio.update(), gradio.update(), gradio.update()
    # Make sure the required_keys have values
    if any(not global_dict[key] for key in required_keys):
        print(f"Missing required value in facefusion.globals")
        return gradio.update(), gradio.update(), gradio.update()
    new_job = JobParams().from_dict(global_dict)

    processors = new_job.frame_processors
    if "face_debugger" in processors:
        processors.remove("face_debugger")
        new_job.frame_processors = processors
    new_job.output_path = normalize_output_path(new_job.source_paths, new_job.target_path, new_job.output_path)
    target_file = gradio.update()
    source_image = gradio.update()
    # If the global dict (without id) is already in the job queue, don't add it again
    for job in JOB_QUEUE:
        if new_job.compare(job):
            print(f"Job already in queue")
            return gradio.update(), target_file, source_image
    new_job.id = len(JOB_QUEUE) + len(COMPLETED_JOBS) + 1  # Add ID field
    temp_test_dir = os.path.join(os.path.dirname(new_job.output_path), "ff_debug")
    if not os.path.exists(temp_test_dir):
        os.makedirs(temp_test_dir)
    job_json_path = os.path.join(temp_test_dir, f"job_{new_job.id}.json")
    with open(job_json_path, "w") as job_json_file:
        job_json_file.write(new_job.to_json())

    print(f"Adding job to queue: {new_job.to_dict()}")
    JOB_QUEUE.append(new_job)
    from facefusion.uis.components.job_queue_options import CLEAR_SOURCE
    target_file = gradio.update(value=None)
    if CLEAR_SOURCE:
        source_image = gradio.update(value=None)

    return gradio.update(value=queue_to_table(), visible=True), target_file, source_image
