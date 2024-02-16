import importlib
import inspect
import os
import pkgutil

import gradio as gr

from facefusion import face_analyser
from facefusion.download import conditional_download
from facefusion.uis.core import load_ui_layout_module
from modules import script_callbacks

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
