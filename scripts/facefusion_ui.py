import importlib
import inspect
import os
import pkgutil
import time
from typing import List, Union
from PIL import Image

import gradio as gr

from facefusion import state_manager
from facefusion.args import apply_args, collect_step_args
from facefusion.core import route
from facefusion.download import conditional_download
from facefusion.filesystem import output_dir, get_output_path_auto
from facefusion.memory import tune_performance
from facefusion.processors.core import get_processors_modules
from facefusion.program import create_program
from facefusion.program_helper import validate_args
from facefusion.uis.core import load_ui_layout_module
from facefusion.workers.core import get_worker_modules
from facefusion.uis.components.instant_runner import create_and_run_job
from facefusion.filesystem import TEMP_DIRECTORY_PATH
from modules import script_callbacks, scripts, scripts_postprocessing

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
                        if model_path.endswith("models"):
                            # Remove the path and the trailing slash
                            model_path = model_path[:-7]
                        model_url = module.MODELS[model]["url"]
                        if "inswapper" in model_url:
                            continue
                        conditional_download(model_path, [model_url])
                # Recursively check for submodules
                find_submodules(module)

    # Start the recursive search from the initial package
    find_submodules(package)


def run_preloads(_, __):
    """
    Run the preloads asynchronously using a ThreadPoolExecutor.
    """
    all_processors = get_processors_modules()
    all_workers = get_worker_modules()

    # Create tasks for all preloads
    for processor in all_processors:
        print(f"Preloading processor {processor.display_name}")
        processor.pre_load()
    for worker in all_workers:
        print(f"Preloading worker {worker.display_name}")
        worker.pre_load()


def load_facefusion():
    from facefusion import logger, globals, state_manager
    globals.output_path = output_dir
    state_manager.init_item('output_path', output_dir)
    program = create_program()
    og_args = vars(program.parse_args())
    program.add_argument_group('processors')
    all_processors = get_processors_modules()
    all_workers = get_worker_modules()
    for processor in all_processors:
        processor.register_args(program)
        processor.apply_args(og_args, state_manager.init_item)
    for worker in all_workers:
        worker.register_args(program)
        worker.apply_args(og_args, state_manager.init_item)

    globals_dict = {}

    if validate_args(program):
        args = vars(program.parse_args())
        ff_args = {key: args[key] for key in args if key not in og_args}
        globals_dict.update(ff_args)
        if state_manager.get_item('command'):
            logger.init(state_manager.get_item('log_level'))
            route(args)

    for key in globals.__dict__:
        if not key.startswith('__') and key not in globals_dict:
            globals_dict[key] = globals.__dict__[key]
            # logger.warn(f"Global variable {key} is not set.", __name__)

    ff_ini = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'facefusion.ini'))
    globals_dict['config_path'] = ff_ini
    # Load the ini file, read each line and set the key-value pair in globals_dict if the value is not None
    with open(ff_ini, 'r') as f:
        for line in f:
            if "=" not in line or line.startswith("#"):
                continue
            key, value = line.strip().split('=')
            if value != 'None' and value != "" and value != "''":
                print(f"Setting {key} to {value} from facefusion.ini")
                globals_dict[key] = value
    apply_args(globals_dict, False)
    state_manager.init_item("config_path", ff_ini)

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
script_callbacks.on_app_started(run_preloads)


def update_source_faces(file_paths: List[str]) -> None:
    if not file_paths:
        print("No source faces provided")
        return

    # Initialize or get the source frame dictionary
    source_dict = state_manager.get_item('source_frame_dict')
    if not source_dict:
        source_dict = {}

    # Update the source paths and dictionary
    source_dict[0] = file_paths  # Use index 0 for primary source faces
    state_manager.set_item('source_paths', file_paths)
    state_manager.set_item('source_frame_dict', source_dict)
    print(f"Updated source_frame_dict: {source_dict}")


def process_internal(is_ff_enabled, image, source_paths=None):
    if not is_ff_enabled:
        print("FaceFusion is disabled")
        return

    if not source_paths or not any(source_paths):
        print("No source faces selected")
        return

    print("FaceFusion is enabled")
    temp_dir = TEMP_DIRECTORY_PATH
    # Ensure the image is in RGB format
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Prepare the image for FaceFusion processing
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    temp_name = f"facefusion_{time.time()}"
    temp_file = os.path.join(temp_dir, f"{temp_name}.jpg")
    image.save(temp_file)
    print(f"FaceFusion processing image: {temp_file}")

    # Set up output directory
    output_dir_path = os.path.join(temp_dir, "output")
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    output_path = os.path.join(output_dir_path, f"{temp_name}_output.jpg")

    # Update source faces in state manager
    update_source_faces(source_paths)

    # Collect arguments from FaceFusion's state
    step_args = collect_step_args()
    step_args['target_path'] = temp_file
    step_args['output_path'] = output_path
    step_args['face_selector_mode'] = 'one'  # Force one face mode for consistent results

    # Create and run the job using instant_runner
    success = create_and_run_job(step_args, keep_state=True)

    # Handle the output from FaceFusion
    if success and os.path.exists(output_path):
        print(f"FaceFusion succeeded: {output_path}")
        with Image.open(output_path) as img:
            result_image = img.copy()
        os.remove(temp_file)
        os.remove(output_path)
        try:
            os.rmdir(output_dir_path)  # Try to remove output dir if empty
        except:
            pass  # Ignore if not empty or other error
        return result_image
    else:
        print("FaceFusion failed")
        return None


class FaceFusionScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.is_ff_enabled = False
        self.source_paths = []

    def title(self):
        return "FaceFusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("FaceFusion", open=False):
            with gr.Row():
                enable = gr.Checkbox(
                    label="Process with FaceFusion",
                    value=False,
                    visible=True,
                )
            with gr.Row():
                source_images = gr.Files(
                    label="Source Face(s)",
                    file_types=["image"],
                    visible=True
                )
        return [enable, source_images]

    def postprocess_image(self, p, pp, enable, source_files):
        self.is_ff_enabled = enable
        self.source_paths = [f.name for f in source_files] if source_files else []
        result_image = process_internal(self.is_ff_enabled, pp.image, self.source_paths)
        if result_image:
            pp.image = result_image


class FaceFusionPostProcessing(scripts_postprocessing.ScriptPostprocessing):
    name = "FaceFusion"
    order = 1999

    def __init__(self):
        super().__init__()
        self.is_ff_enabled = False
        self.source_paths = []

    def ui(self):
        with gr.Accordion("FaceFusion", open=False):
            with gr.Row():
                enable = gr.Checkbox(
                    label="Process with FaceFusion",
                    value=False,
                    visible=True,
                )
            with gr.Row():
                source_images = gr.Files(
                    label="Source Face(s)",
                    file_types=["image"],
                    visible=True
                )
        return {
            "is_ff_enabled": enable,
            "source_files": source_images
        }

    def process(self, pp, *, is_ff_enabled, source_files):
        self.is_ff_enabled = is_ff_enabled
        self.source_paths = [f.name for f in source_files] if source_files else []
        result_image = process_internal(self.is_ff_enabled, pp.image, self.source_paths)
        if result_image:
            pp.image = result_image
