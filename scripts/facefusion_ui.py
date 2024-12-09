import importlib
import inspect
import os
import pkgutil
import signal

import gradio as gr

from facefusion.args import apply_args
from facefusion.core import route
from facefusion.download import conditional_download
from facefusion.exit_helper import graceful_exit
from facefusion.memory import tune_performance
from facefusion.program import create_program
from facefusion.program_helper import validate_args
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


def load_facefusion():
    import facefusion
    from facefusion import logger, globals, state_manager, args

    signal.signal(signal.SIGINT, lambda signal_number, frame: graceful_exit(0))
    program = create_program()

    if validate_args(program):
        args = vars(program.parse_args())
        #apply_args(args, False)

        if state_manager.get_item('command'):
            logger.init(state_manager.get_item('log_level'))
            route(args)
        else:
            program.print_help()

    globals_dict = {}
    for key in globals.__dict__:
        if not key.startswith('__'):
            globals_dict[key] = globals.__dict__[key]

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

    #run_pre_checks(facefusion)
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

