import gradio

from facefusion import state_manager
from facefusion.uis.components import (age_modifier_options, common_options, expression_restorer_options, \
                                       face_debugger_options, face_detector, face_editor_options, \
                                       face_enhancer_options, face_landmarker, face_masker, face_selector,
                                       face_swapper_options, frame_colorizer_options, \
                                       frame_enhancer_options, instant_runner, job_manager, job_runner,
                                       lip_syncer_options, style_changer_options, \
                                       style_transfer_options, output, \
                                       output_options, \
                                       preview, processors, source, target, temp_frame, trim_frame, execution,
                                       execution_thread_count, execution_queue_count, ui_workflow)


def pre_check() -> bool:
    return True


def render() -> gradio.Blocks:
    with gradio.Blocks() as layout:
        with gradio.Row():
            with gradio.Column(scale=4):
                with gradio.Blocks():
                    processors.render()
                with gradio.Blocks():
                    age_modifier_options.render()
                with gradio.Blocks():
                    expression_restorer_options.render()
                with gradio.Blocks():
                    face_debugger_options.render()
                with gradio.Blocks():
                    face_editor_options.render()
                with gradio.Blocks():
                    face_enhancer_options.render()
                with gradio.Blocks():
                    face_swapper_options.render()
                with gradio.Blocks():
                    frame_colorizer_options.render()
                with gradio.Blocks():
                    frame_enhancer_options.render()
                with gradio.Blocks():
                    lip_syncer_options.render()
                with gradio.Blocks():
                    style_changer_options.render()
                with gradio.Blocks():
                    execution.render()
                    execution_thread_count.render()
                    execution_queue_count.render()
                # with gradio.Blocks():
                #     memory.render()
                with gradio.Blocks():
                    temp_frame.render()
                with gradio.Blocks():
                    output_options.render()
                with gradio.Blocks():
                    common_options.render()
                with gradio.Blocks():
                    output.render()
                with gradio.Blocks():
                    ui_workflow.render()
                    instant_runner.render()
                    job_runner.render()
                    job_manager.render()
                # with gradio.Blocks():
                #     terminal.render()
            with gradio.Column(scale=4):
                with gradio.Blocks() as source_block:
                    source.render()
                with gradio.Blocks():
                    style_transfer_options.render()
                with gradio.Blocks():
                    target.render()

            with gradio.Column(scale=7):
                with gradio.Blocks():
                    preview.render()
                with gradio.Blocks():
                    trim_frame.render()
                with gradio.Blocks():
                    face_selector.render()
                with gradio.Blocks():
                    face_masker.render()
                with gradio.Blocks():
                    face_detector.render()
                with gradio.Blocks():
                    face_landmarker.render()

    return layout


def listen() -> None:
    processors.listen()
    age_modifier_options.listen()
    expression_restorer_options.listen()
    face_debugger_options.listen()
    face_editor_options.listen()
    face_enhancer_options.listen()
    face_swapper_options.listen()
    frame_colorizer_options.listen()
    frame_enhancer_options.listen()
    lip_syncer_options.listen()
    style_changer_options.listen()
    style_transfer_options.listen()
    execution.listen()
    execution_thread_count.listen()
    execution_queue_count.listen()
    # memory.listen()
    temp_frame.listen()
    output_options.listen()
    source.listen()
    target.listen()
    output.listen()
    #ui_workflow.listen()
    instant_runner.listen()
    job_runner.listen()
    job_manager.listen()
    # terminal.listen()
    preview.listen()
    trim_frame.listen()
    face_selector.listen()
    face_masker.listen()
    face_detector.listen()
    face_landmarker.listen()
    common_options.listen()


def run(ui: gradio.Blocks) -> None:
    ui.launch(favicon_path='facefusion.ico', inbrowser=state_manager.get_item('open_browser'))
