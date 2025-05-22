import argparse
import logging
import os
import sys

# Disable httpcore.http11 logging
logging.getLogger('httpcore').setLevel(logging.WARNING)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ext_dir = os.path.join(base_dir, 'extensions', 'sd_facefusion')

if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)
else:
    print(f"Ext dir already in path: {ext_dir}")


def preload(parser: argparse.ArgumentParser):
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    # Ensure extension_sd_facefusion.facefusion is added to the PATH as 'facefusion'
    os.environ['PATH'] = os.pathsep.join(
        [os.environ['PATH'], os.path.abspath(os.path.join(os.path.dirname(__file__), 'facefusion'))])
    print(f"Preloading globals to state_manager...")
    try:
        from facefusion import globals, state_manager, args
    except ImportError:
        from extensions.sd_facefusion.facefusion import globals, state_manager, args

    globals_dict = {}
    for key in globals.__dict__:
        if not key.startswith('__'):
            globals_dict[key] = globals.__dict__[key]

    ff_ini = os.path.abspath(os.path.join(os.path.dirname(__file__), 'facefusion.ini'))
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
    args.apply_args(globals_dict, False)
    state_manager.init_item("config_path", ff_ini)
    
    # Initialize YOLO mask state items if not already set
    if state_manager.get_item('custom_yolo_model') is None:
        state_manager.init_item('custom_yolo_model', None)
    if state_manager.get_item('custom_yolo_confidence') is None:
        state_manager.init_item('custom_yolo_confidence', 0.5)
    if state_manager.get_item('custom_yolo_radius') is None:
        state_manager.init_item('custom_yolo_radius', 10)
    # Initialize mask time state items
    if state_manager.get_item('mask_disabled_times') is None:
        state_manager.init_item('mask_disabled_times', [0])
    if state_manager.get_item('mask_enabled_times') is None:
        state_manager.init_item('mask_enabled_times', [])
