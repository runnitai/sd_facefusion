import importlib
import inspect
import os
import pkgutil
import time
from typing import Union, List

import gradio
import gradio as gr
from PIL import Image
from transformers.image_transforms import to_pil_image

from facefusion import face_analyser, wording, globals
from facefusion.processors.frame import choices as frame_processors_choices, globals as frame_processors_globals
from facefusion.download import conditional_download
from facefusion.filesystem import TEMP_DIRECTORY_PATH
from facefusion.job_params import JobParams
from facefusion.memory import tune_performance
from facefusion.uis.components.output import start_job
from facefusion.uis.core import load_ui_layout_module
from modules import script_callbacks, scripts, scripts_postprocessing, ui_components

# export CUDA_MODULE_LOADING=LAZY
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'


def run_pre_checks(package):
    # Recursively find all submodules in a given package
    def find_submodules(package):
        if hasattr(package, '__path__'):
            for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
                # Import the module
                module = importlib.import_module(modname)

                # Check if the module has a 'pre_check' function
                if hasattr(module, 'pre_check') and inspect.isfunction(module.pre_check):
                    # Run the 'pre_check' function
                    module.pre_check()

                if hasattr(module, "MODELS"):
                    for model in module.MODELS:
                        model_path = os.path.dirname(module.MODELS[model]["path"])
                        print(f"Checking for model: {model} at {model_path}")
                        model_url = module.MODELS[model]["url"]
                        conditional_download(model_path, [model_url])
                # Recursively check for submodules
                find_submodules(module)

    # Start the recursive search from the initial package
    find_submodules(package)


def load_facefusion():
    import facefusion
    run_pre_checks(facefusion)
    face_analyser.pre_check()
    tune_performance()
    with gr.Blocks() as ff_ui:
        with gr.Tabs():
            with gr.Tab(label="File"):
                default_layout = load_ui_layout_module("default")
                default_layout.render()
                default_layout.listen()
            with gr.Tab(label="Live"):
                live_layout = load_ui_layout_module("webcam")
                live_layout.render()
                live_layout.listen()
            with gr.Tab(label="Benchmark"):
                bench_layout = load_ui_layout_module("benchmark")
                bench_layout.render()
                bench_layout.listen()
        return ((ff_ui, "RD FaceFusion", "ff_ui_clean"),)


script_callbacks.on_ui_tabs(load_facefusion)

__version__ = "0.0.1"


def process_internal(is_ff_enabled, image, source_paths: List[str], face_selector_mode="one",
                     frame_processors=["face_swapper"],
                     face_swapper_model="inswapper_128_fp16", face_enhancer_model="gfpgan_1.4",
                     frame_enhancer_model="real_esrgan_x4plus"):
    if not is_ff_enabled:
        return

    if not len(source_paths):
        print("No source images provided")
        return

    temp_dir = TEMP_DIRECTORY_PATH
    # Ensure the image is in RGB format
    if not isinstance(image, Image.Image):
        image = to_pil_image(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Prepare the image for FaceFusion processing
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    temp_name = f"facefusion_{time.time()}"
    temp_file = os.path.join(temp_dir, f"{temp_name}.jpg")
    image.save(temp_file)
    ff_params = JobParams.from_globals()
    ff_params.target_path = temp_file
    ff_params.source_paths = source_paths
    ff_params.output_image_quality = 100
    ff_params.temp_frame_quality = 100
    ff_params.face_selector_mode = face_selector_mode
    ff_params.frame_processors = frame_processors
    frame_processors_globals.face_swapper_model = face_swapper_model
    frame_processors_globals.face_enhancer_model = face_enhancer_model
    frame_processors_globals.frame_enhancer_model = frame_enhancer_model
    out_path = start_job(ff_params)

    # Handle the output from FaceFusion
    if out_path and os.path.exists(out_path):
        with Image.open(out_path) as img:
            result_image = img.copy()
        os.remove(temp_file)
        os.remove(out_path)
        return result_image
    else:
        print("FaceFusion failed")
        return None


class FaceFusionScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.is_ff_enabled = False
        self.selector_mode = "one"
        self.sources = []
        self.frame_processors = ["face_swapper"]
        self.face_swapper_model = frame_processors_globals.face_swapper_model
        self.face_enhancer_model = frame_processors_globals.face_enhancer_model
        self.frame_enhancer_model = frame_processors_globals.frame_enhancer_model

    def __repr__(self):
        return f"{self.__class__.__name__}(version={__version__})"

    def title(self):
        return "FaceFusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        enable, selector_mode, sources, processors, swapper_model, face_enhancer_model, frame_enhancer_model = ui_internal(self)
        return [enable, selector_mode, sources, processors,swapper_model, face_enhancer_model, frame_enhancer_model]

    def postprocess_image(self, p, pp, *args_):
        result_image = process_internal(self.is_ff_enabled, pp.image, self.sources, self.selector_mode,
                                        self.frame_processors, self.face_swapper_model, self.face_enhancer_model,
                                        self.frame_enhancer_model)
        if result_image:
            pp.image = result_image


class FaceFusionPostProcessing(scripts_postprocessing.ScriptPostprocessing):
    name = "FaceFusion"
    order = 1999

    def __init__(self):
        super().__init__()
        self.is_ff_enabled = False
        self.selector_mode = "one"
        self.sources = []
        self.frame_processors = ["face_swapper"]
        self.face_swapper_model = frame_processors_globals.face_swapper_model
        self.face_enhancer_model = frame_processors_globals.face_enhancer_model
        self.frame_enhancer_model = frame_processors_globals.frame_enhancer_model

    def ui(self):
        enable, selector_mode, sources, processors, swapper_model, face_enhancer_model, frame_enhancer_model = ui_internal(self)
        return {
            "is_ff_enabled": enable,
            "facefusion_selector_mode": selector_mode,
            "facefusion_sources": sources,
            "frame_processors": processors,
            "face_swapper_model": swapper_model,
            "face_enhancer_model": face_enhancer_model,
            "frame_enhancer_model": frame_enhancer_model,
        }

    def process_firstpass(self, pp, **args):
        # This method can be used to set any preliminary flags or checks before the main processing
        pass

    def process(self, pp, **args):
        result_image = process_internal(self.is_ff_enabled, pp.image, self.sources, self.selector_mode,
                                        self.frame_processors,
                                        self.face_swapper_model, self.face_enhancer_model, self.frame_enhancer_model)
        if result_image:
            pp.image = result_image

    def image_changed(self):
        pass  # Implement if needed to handle image change events


def ui_internal(script_cls: Union[FaceFusionScript, FaceFusionPostProcessing]):
    accordion_title = "FaceFusion"
    checkbox_label = "Process with FaceFusion"
    if isinstance(script_cls, FaceFusionPostProcessing):
        with ui_components.InputAccordion(False, label="FaceFusion") as ff_enable:
            with gr.Column(variant="panel", visible=False) as settings_row:
                with gr.Row():
                    frame_processors = gr.CheckboxGroup(
                        label="Frame Processors",
                        choices=["face_swapper", "face_enhancer", "frame_enhancer"],
                        value=["face_swapper"]
                    )
                with gr.Row():
                    selector_mode = gr.Radio(
                        label="Face Selector Mode",
                        choices=["one", "many"],
                        default="one",
                        value="one",
                        elem_id="facefusion_selector_mode",
                    )
                with gr.Row():
                    swapper_model = gradio.Dropdown(
                        label=wording.get('uis.swapper_model'),
                        choices=frame_processors_choices.face_swapper_models,
                        value=frame_processors_globals.face_swapper_model,
                        visible='face_swapper' in globals.frame_processors
                    )
                with gr.Row():
                    face_enhancer_model = gradio.Dropdown(
                        label=wording.get('uis.face_enhancer_model'),
                        choices=frame_processors_choices.face_enhancer_models,
                        value=frame_processors_globals.face_enhancer_model,
                        visible='face_enhancer' in globals.frame_processors,
                        elem_id='face_enhancer_model'
                    )
                with gr.Row():
                    frame_enhancer_model = gradio.Dropdown(
                        label=wording.get('uis.frame_enhancer_model'),
                        choices=frame_processors_choices.frame_enhancer_models,
                        value=frame_processors_globals.frame_enhancer_model,
                        visible=False,
                        elem_id='frame_enhancer_model'
                    )
                with gr.Row():
                    source_files = gr.Files(
                        label="Source Images",
                        interactive=True,
                        file_types=
                        [
                            '.png',
                            '.jpg',
                            '.webp'
                        ],
                        elem_id="facefusion_sources",
                    )
                    source_image = gr.Image(
                        label="Source Images",
                        interactive=False,
                        visible=False,
                        type="filepath",
                        elem_id="facefusion_sources",
                    )
    else:
        with gr.Accordion(accordion_title, open=False):
            ff_enable = gr.Checkbox(
                label=checkbox_label,
                value=False,
                visible=True,
            )
            with gr.Column(variant="panel", visible=False) as settings_row:
                with gr.Row():
                    frame_processors = gr.CheckboxGroup(
                        label="Frame Processors",
                        choices=["face_swapper", "face_enhancer", "frame_enhancer"],
                        value=["face_swapper"]
                    )
                with gr.Row():
                    selector_mode = gr.Radio(
                        label="Face Selector Mode",
                        choices=["one", "many"],
                        default="one",
                        elem_id="facefusion_selector_mode",
                    )
                with gr.Row():
                    swapper_model = gradio.Dropdown(
                        label="Face Swapper Model",
                        choices=frame_processors_choices.face_swapper_models,
                        value=frame_processors_globals.face_swapper_model,
                        visible='face_swapper' in globals.frame_processors
                    )
                with gr.Row():
                    face_enhancer_model = gradio.Dropdown(
                        label="Face Enhancer Model",
                        choices=frame_processors_choices.face_enhancer_models,
                        value=frame_processors_globals.face_enhancer_model,
                        visible='face_enhancer' in globals.frame_processors,
                        elem_id='face_enhancer_model'
                    )
                with gr.Row():
                    frame_enhancer_model = gradio.Dropdown(
                        label="Frame Enhancer Model",
                        choices=frame_processors_choices.frame_enhancer_models,
                        value=frame_processors_globals.frame_enhancer_model,
                        visible=False,
                        elem_id='frame_enhancer_model'
                    )
                with gr.Row():
                    source_files = gr.Files(
                        label="Source Images",
                        interactive=True,
                        file_types=
                        [
                            '.png',
                            '.jpg',
                            '.webp'
                        ],
                        elem_id="facefusion_sources",
                    )
                    source_image = gr.Image(
                        label="Source Images",
                        interactive=False,
                        type="filepath",
                        visible=False,
                        elem_id="facefusion_sources",
                    )

    def update_sources(files):

        file_names = [file.name for file in files] if files else []
        setattr(script_cls, "sources", file_names)
        if len(file_names):
            return gr.update(value=file_names[0], visible=True)
        return gr.update(value=None, visible=False)

    def toggle_settings(enable):
        setattr(script_cls, "is_ff_enabled", enable)
        return gr.update(visible=enable)

    def toggle_dropdowns(processors):
        show_swap_model = 'face_swapper' in processors
        show_face_model = 'face_enhancer' in processors
        show_frame_model = 'frame_enhancer' in processors
        return gr.update(visible=show_swap_model), gr.update(visible=show_face_model), gr.update(visible=show_frame_model)

    ff_enable.change(fn=toggle_settings, inputs=[ff_enable], outputs=[settings_row])
    source_files.change(fn=update_sources, inputs=[source_files], outputs=[source_image])
    frame_processors.change(fn=toggle_dropdowns, inputs=[frame_processors], outputs=[swapper_model, face_enhancer_model, frame_enhancer_model])

    selector_mode.change(fn=lambda mode: setattr(script_cls, "selector_mode", mode), inputs=[selector_mode])

    swapper_model.change(
        fn=lambda model: setattr(frame_processors_globals, "face_swapper_model", model),
        inputs=[swapper_model]
    )
    face_enhancer_model.change(
        fn=lambda model: setattr(frame_processors_globals, "face_enhancer_model", model),
        inputs=[face_enhancer_model]
    )
    frame_enhancer_model.change(
        fn=lambda model: setattr(frame_processors_globals, "frame_enhancer_model", model),
        inputs=[frame_enhancer_model]
    )

    return ff_enable, selector_mode, source_files, frame_processors, swapper_model, face_enhancer_model, frame_enhancer_model
