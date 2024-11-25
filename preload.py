import argparse
import os
import logging

# Disable httpcore.http11 logging
logging.getLogger('httpcore').setLevel(logging.WARNING)


def preload(parser: argparse.ArgumentParser):
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
