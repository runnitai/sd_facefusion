import traceback

import gradio

from facefusion.uis.components import frame_processors, frame_processors_options, execution, \
    temp_frame, output_options, source, \
    target, output, preview, trim_frame, face_analyser, face_selector, job_queue, job_queue_options, face_masker


def pre_check() -> bool:
    return True


def pre_render() -> bool:
    return True


def render() -> gradio.Blocks:
    with gradio.Blocks() as layout:
        with gradio.Row():
            with gradio.Column(scale=2):
                with gradio.Blocks():
                    frame_processors.render()
                    frame_processors_options.render()
                with gradio.Blocks():
                    execution.render()
                with gradio.Blocks():
                    temp_frame.render()
                with gradio.Blocks():
                    output_options.render()
                    job_queue.render()
                    output.render()
                    job_queue_options.render()
            with gradio.Column(scale=2):
                with gradio.Blocks():
                    source.render()
                with gradio.Blocks():
                    target.render()
            with gradio.Column(scale=3):
                with gradio.Blocks():
                    preview.render()
                    trim_frame.render()
                with gradio.Blocks():
                    face_selector.render()
                with gradio.Blocks():
                    face_masker.render()
                with gradio.Blocks():
                    face_analyser.render()

    return layout


def listen() -> None:
    try:
        frame_processors.listen()
        frame_processors_options.listen()
        execution.listen()
        temp_frame.listen()
        output_options.listen()
        source.listen()
        target.listen()
        preview.listen()
        trim_frame.listen()
        face_selector.listen()
        face_masker.listen()
        face_analyser.listen()
        job_queue.listen()
        job_queue_options.listen()
        output.listen()
    except Exception as e:
        print(f"Exception listening to output component: {e}")
        traceback.print_exc()


def run(ui: gradio.Blocks) -> None:
    ui.launch(show_api=False, quiet=True)
