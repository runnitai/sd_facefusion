import os
from typing import Any, Dict
from functools import lru_cache
import threading
import cv2
import numpy
import onnxruntime
from tqdm import tqdm

import facefusion.globals
from facefusion import wording
from facefusion.typing import VisionFrame, ModelValue, Fps
from facefusion.execution import apply_execution_provider_options
from facefusion.vision import get_video_frame, count_video_frame_total, read_image, detect_video_fps
from facefusion.filesystem import resolve_relative_path
from facefusion.download import conditional_download

CONTENT_ANALYSER = None
THREAD_LOCK: threading.Lock = threading.Lock()
MODELS: Dict[str, ModelValue] = \
    {
        'open_nsfw':
            {
                'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/open_nsfw.onnx',
                'path': resolve_relative_path('../.assets/models/open_nsfw.onnx')
            }
    }
PROBABILITY_LIMIT = 0.80
RATE_LIMIT = 5
STREAM_COUNTER = 0
DEBUG = os.environ.get('SKIP_PREDICTOR', False)


def get_content_analyser() -> Any:
    if DEBUG:
        return None
    global CONTENT_ANALYSER

    with THREAD_LOCK:
        if CONTENT_ANALYSER is None:
            model_path = MODELS.get('open_nsfw').get('path')
            CONTENT_ANALYSER = onnxruntime.InferenceSession(model_path, providers=apply_execution_provider_options(
                facefusion.globals.execution_providers))
    return CONTENT_ANALYSER


def clear_content_analyser() -> None:
    global CONTENT_ANALYSER

    CONTENT_ANALYSER = None


def pre_check() -> bool:
    if DEBUG:
        return True
    if not facefusion.globals.skip_download:
        download_directory_path = resolve_relative_path('../.assets/models')
        model_url = MODELS.get('open_nsfw').get('url')
        conditional_download(download_directory_path, [model_url])
    return True


def analyse_stream(frame: VisionFrame, video_fps: Fps) -> bool:
    if DEBUG:
        return False
    global STREAM_COUNTER

    STREAM_COUNTER = STREAM_COUNTER + 1
    if STREAM_COUNTER % int(video_fps) == 0:
        return analyse_frame(frame)
    return False


def prepare_frame(frame: VisionFrame) -> VisionFrame:
    frame = cv2.resize(frame, (224, 224)).astype(numpy.float32)
    frame -= numpy.array([104, 117, 123]).astype(numpy.float32)
    frame = numpy.expand_dims(frame, axis=0)
    return frame


def analyse_frame(frame: VisionFrame) -> bool:
    if DEBUG:
        return False
    content_analyser = get_content_analyser()
    frame = prepare_frame(frame)
    probability = content_analyser.run(None,
                                       {
                                           'input:0': frame
                                       })[0][0][1]
    return probability > PROBABILITY_LIMIT


@lru_cache(maxsize=None)
def analyse_image(image_path: str) -> bool:
    if DEBUG:
        return False
    frame = read_image(image_path)
    return analyse_frame(frame)


@lru_cache(maxsize=None)
def analyse_video(video_path: str, start_frame: int, end_frame: int) -> bool:
    if DEBUG:
        return False
    video_frame_total = count_video_frame_total(video_path)
    video_fps = detect_video_fps(video_path)
    frame_range = range(start_frame or 0, end_frame or video_frame_total)
    rate = 0.0
    counter = 0
    with tqdm(total=len(frame_range), desc=wording.get('analysing'), unit='frame', ascii=' =',
              disable=facefusion.globals.log_level in ['warn', 'error']) as progress:
        for frame_number in frame_range:
            if frame_number % int(video_fps) == 0:
                frame = get_video_frame(video_path, frame_number)
                if analyse_frame(frame):
                    counter += 1
            rate = counter * int(video_fps) / len(frame_range) * 100
            progress.update()
            progress.set_postfix(rate=rate)
    return rate > RATE_LIMIT
