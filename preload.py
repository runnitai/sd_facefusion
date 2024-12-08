import argparse
import os
import logging
import signal

from facefusion import logger
from facefusion.args import apply_args
from facefusion.core import route
from facefusion.exit_helper import graceful_exit
from facefusion.program import create_program
from facefusion.program_helper import validate_args

# Disable httpcore.http11 logging
logging.getLogger('httpcore').setLevel(logging.WARNING)


def preload(parser: argparse.ArgumentParser):
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    print(f"Preloading globals to state_manager...")
    from facefusion import globals, state_manager, args
    # signal.signal(signal.SIGINT, lambda signal_number, frame: graceful_exit(0))
    # program = create_program()
    #
    # if validate_args(program):
    #     args = vars(program.parse_args())
    #     apply_args(args, state_manager.init_item)
    #
    #     if state_manager.get_item('command'):
    #         logger.init(state_manager.get_item('log_level'))
    #         route(args)
    #     else:
    #         program.print_help()

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


