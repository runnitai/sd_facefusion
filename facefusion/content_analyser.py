import os
from functools import lru_cache

import cv2
import numpy

from facefusion import inference_manager
from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.typing import VisionFrame, Fps, ModelSet, InferencePool, ModelOptions

MODEL_SET: ModelSet = \
    {
        'open_nsfw':
            {
                'hashes':
                    {
                        'content_analyser':
                            {
                                'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/open_nsfw.hash',
                                'path': resolve_relative_path('../.assets/models/open_nsfw.hash')
                            }
                    },
                'sources':
                    {
                        'content_analyser':
                            {
                                'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/open_nsfw.onnx',
                                'path': resolve_relative_path('../.assets/models/open_nsfw.onnx')
                            }
                    },
                'size': (224, 224),
                'mean': [104, 117, 123]
            }
    }
PROBABILITY_LIMIT = 0.80
RATE_LIMIT = 10
STREAM_COUNTER = 0
DEBUG = os.environ.get('SKIP_PREDICTOR', True)


def get_inference_pool() -> InferencePool:
    model_sources = get_model_options().get('sources')
    return inference_manager.get_inference_pool(__name__, model_sources)


def clear_inference_pool() -> None:
    inference_manager.clear_inference_pool(__name__)


def get_model_options() -> ModelOptions:
    return MODEL_SET.get('open_nsfw')


def pre_check() -> bool:
    return True


def analyse_stream(vision_frame: VisionFrame, video_fps: Fps) -> bool:
    return False


def analyse_frame(vision_frame: VisionFrame) -> bool:
    return False


def forward(vision_frame: VisionFrame) -> float:
    content_analyser = get_inference_pool().get('content_analyser')

    with conditional_thread_semaphore():
        probability = content_analyser.run(None,
                                           {
                                               'input': vision_frame
                                           })[0][0][1]

    return probability


def prepare_frame(vision_frame: VisionFrame) -> VisionFrame:
    model_size = get_model_options().get('size')
    model_mean = get_model_options().get('mean')
    vision_frame = cv2.resize(vision_frame, model_size).astype(numpy.float32)
    vision_frame -= numpy.array(model_mean).astype(numpy.float32)
    vision_frame = numpy.expand_dims(vision_frame, axis=0)
    return vision_frame


@lru_cache(maxsize=None)
def analyse_image(image_path: str) -> bool:
    return False


@lru_cache(maxsize=None)
def analyse_video(video_path: str, start_frame: int, end_frame: int) -> bool:
    return False
