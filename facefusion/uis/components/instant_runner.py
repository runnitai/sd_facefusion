import os
from time import sleep
from typing import Optional, Tuple

import gradio

from facefusion import process_manager, state_manager, wording
from facefusion.args import collect_step_args
from facefusion.core import process_step
from facefusion.ff_status import FFStatus
from facefusion.filesystem import is_directory, is_image, is_video, get_output_path_auto
from facefusion.jobs import job_helper, job_manager, job_runner, job_store
from facefusion.temp_helper import clear_temp_directory
from facefusion.typing import Args, UiWorkflow
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.uis.ui_helper import suggest_output_path

INSTANT_RUNNER_WRAPPER: Optional[gradio.Row] = None
INSTANT_RUNNER_START_BUTTON: Optional[gradio.Button] = None
INSTANT_RUNNER_STOP_BUTTON: Optional[gradio.Button] = None
INSTANT_RUNNER_CLEAR_BUTTON: Optional[gradio.Button] = None


def render() -> None:
    global INSTANT_RUNNER_WRAPPER
    global INSTANT_RUNNER_START_BUTTON
    global INSTANT_RUNNER_STOP_BUTTON
    global INSTANT_RUNNER_CLEAR_BUTTON

    if job_manager.init_jobs(state_manager.get_item('jobs_path')):
        is_instant_runner = state_manager.get_item('ui_workflow') == 'instant_runner'

        with gradio.Row(visible=is_instant_runner) as INSTANT_RUNNER_WRAPPER:
            INSTANT_RUNNER_START_BUTTON = gradio.Button(
                value=wording.get('uis.start_button'),
                variant='primary',
                size='sm'
            )
            INSTANT_RUNNER_STOP_BUTTON = gradio.Button(
                value=wording.get('uis.stop_button'),
                variant='primary',
                size='sm',
                visible=False
            )
            INSTANT_RUNNER_CLEAR_BUTTON = gradio.Button(
                value=wording.get('uis.clear_button'),
                size='sm'
            )

        register_ui_component('instant_runner_start_button', INSTANT_RUNNER_START_BUTTON)
        register_ui_component('instant_runner_stop_button', INSTANT_RUNNER_STOP_BUTTON)


def listen() -> None:
    output_image = get_ui_component('output_image')
    output_video = get_ui_component('output_video')
    ui_workflow_dropdown = get_ui_component('ui_workflow_dropdown')

    if output_image and output_video:
        INSTANT_RUNNER_START_BUTTON.click(start, _js='start_status', outputs=[INSTANT_RUNNER_START_BUTTON, INSTANT_RUNNER_STOP_BUTTON])
        INSTANT_RUNNER_START_BUTTON.click(run, outputs=[INSTANT_RUNNER_START_BUTTON, INSTANT_RUNNER_STOP_BUTTON,
                                                        output_image, output_video])
        INSTANT_RUNNER_STOP_BUTTON.click(stop, _js='stop_status', outputs=[INSTANT_RUNNER_START_BUTTON, INSTANT_RUNNER_STOP_BUTTON])
        INSTANT_RUNNER_CLEAR_BUTTON.click(clear, _js='stop_status', outputs=[output_image, output_video])
    if ui_workflow_dropdown:
        ui_workflow_dropdown.change(remote_update, inputs=ui_workflow_dropdown, outputs=INSTANT_RUNNER_WRAPPER)


def remote_update(ui_workflow: UiWorkflow) -> gradio.update:
    is_instant_runner = ui_workflow == 'instant_runner'

    return gradio.update(visible=is_instant_runner)


def start() -> Tuple[gradio.update, gradio.update]:
    while not process_manager.is_processing():
        sleep(0.5)
    return gradio.update(visible=False), gradio.update(visible=True)


def run() -> Tuple[gradio.update, gradio.update, gradio.update, gradio.update]:
    status = FFStatus()
    target_path = state_manager.get_item('target_path')
    target_file = os.path.basename(target_path)
    if len(target_file) > 50:
        truncated_target_base_name = f"{target_file[:20]}...{target_file[-20:]}"
    else:
        truncated_target_base_name = target_file

    status.start(f"Processing target file: {truncated_target_base_name}")

    step_args = collect_step_args()
    output_path = get_output_path_auto()
    step_args['output_path'] = output_path
    if is_directory(step_args.get('output_path')):
        step_args['output_path'] = suggest_output_path(output_path,
                                                       state_manager.get_item('target_path'))
    if job_manager.init_jobs(state_manager.get_item('jobs_path')):
        create_and_run_job(step_args)
        state_manager.set_item('output_path', output_path)
        status.finish(f"Finished processing target file: {truncated_target_base_name}")
    if is_image(step_args.get('output_path')):
        return gradio.update(visible=True), gradio.update(visible=False), gradio.update(
            value=step_args.get('output_path'), visible=True), gradio.update(value=None, visible=False)
    if is_video(step_args.get('output_path')):
        return gradio.update(visible=True), gradio.update(visible=False), gradio.update(value=None,
                                                                                        visible=False), gradio.update(
            value=step_args.get('output_path'), visible=True)
    return gradio.update(visible=True), gradio.update(visible=False), gradio.update(value=None), gradio.update(
        value=None)


def create_and_run_job(step_args: Args) -> bool:
    job_id = job_helper.suggest_job_id('ui')

    for key in job_store.get_job_keys():
        state_manager.sync_item(key)  # type:ignore

    return job_manager.create_job(job_id) and job_manager.add_step(job_id, step_args) and job_manager.submit_job(
        job_id) and job_runner.run_job(job_id, process_step)


def stop() -> Tuple[gradio.update, gradio.update]:
    process_manager.stop()
    status = FFStatus()
    status.finish(f"Stopped processing target file: {state_manager.get_item('target_path')}")
    return gradio.update(visible=True), gradio.update(visible=False)


def clear() -> Tuple[gradio.update, gradio.update]:
    while process_manager.is_processing():
        sleep(0.5)
    if state_manager.get_item('target_path'):
        clear_temp_directory(state_manager.get_item('target_path'))
    status = FFStatus()
    status.finish(f"Cleared target file: {state_manager.get_item('target_path')}")
    return gradio.update(value=None), gradio.update(value=None)
