from typing import Tuple

import numpy

from facefusion.face_helper import warp_face_by_face_landmark_5
from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.typing import Embedding, FaceLandmark5, ModelSet, VisionFrame
from facefusion.workers.base_worker import BaseWorker


class FaceRecognizer(BaseWorker):
    MODEL_SET: ModelSet = \
        {
            'arcface':
                {
                    'hashes':
                        {
                            'face_recognizer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.hash',
                                    'path': resolve_relative_path('../.assets/models/arcface_w600k_r50.hash')
                                }
                        },
                    'sources':
                        {
                            'face_recognizer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.onnx',
                                    'path': resolve_relative_path('../.assets/models/arcface_w600k_r50.onnx')
                                }
                        },
                    'template': 'arcface_112_v2',
                    'size': (112, 112)
                }
        }

    default_model = 'arcface'
    model_key = None
    multi_model = False
    preload = True
    preferred_provider = 'cuda'

    def forward(self, crop_vision_frame: VisionFrame) -> Embedding:
        face_recognizer = self.get_inference_pool().get('face_recognizer')

        with conditional_thread_semaphore():
            embedding = face_recognizer.run(None,
                                            {
                                                'input': crop_vision_frame
                                            })[0]

        return embedding

    def calc_embedding(self, temp_vision_frame: VisionFrame, face_landmark_5: FaceLandmark5) -> Tuple[
        Embedding, Embedding]:
        model_template = self.get_model_options().get('template')
        model_size = self.get_model_options().get('size')
        crop_vision_frame, matrix = warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, model_template,
                                                                 model_size)
        crop_vision_frame = crop_vision_frame / 127.5 - 1
        crop_vision_frame = crop_vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
        crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0)
        embedding = self.forward(crop_vision_frame)
        embedding = embedding.ravel()
        normed_embedding = embedding / numpy.linalg.norm(embedding)
        return embedding, normed_embedding

