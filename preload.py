import argparse
import os


def preload(parser: argparse.ArgumentParser):
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
