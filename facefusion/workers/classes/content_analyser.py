import os
from functools import lru_cache

import cv2
import numpy

from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.typing import VisionFrame, Fps, ModelSet, ModelOptions
from facefusion.workers.base_worker import BaseWorker


class ContentAnalyser(BaseWorker):
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

    default_model = 'open_nsfw'
    model_key = None
    multi_model = False

    def get_model_options(self) -> ModelOptions:
        return self.MODEL_SET.get(self.default_model)

    def forward(self, vision_frame: VisionFrame) -> float:
        content_analyser = self.get_inference_pool().get('content_analyser')

        with conditional_thread_semaphore():
            probability = content_analyser.run(None,
                                               {
                                                   'input': vision_frame
                                               })[0][0][1]

        return probability

    def pre_check(self) -> bool:
        return True

    def analyse_stream(self,
                       vision_frame: VisionFrame, video_fps: Fps) -> bool:
        return False

    def analyse_frame(self, vision_frame: VisionFrame) -> bool:
        return False

    def prepare_frame(self, vision_frame: VisionFrame) -> VisionFrame:
        model_size = self.get_model_options().get('size')
        model_mean = self.get_model_options().get('mean')
        vision_frame = cv2.resize(vision_frame, model_size).astype(numpy.float32)
        vision_frame -= numpy.array(model_mean).astype(numpy.float32)
        vision_frame = numpy.expand_dims(vision_frame, axis=0)
        return vision_frame

    @lru_cache(maxsize=None)
    def analyse_image(self, image_path: str) -> bool:
        return False

    @lru_cache(maxsize=None)
    def analyse_video(self, video_path: str, start_frame: int, end_frame: int) -> bool:
        return False
