from typing import Tuple

import cv2
import numpy

from facefusion import state_manager
from facefusion.face_helper import create_rotated_matrix_and_size, estimate_matrix_by_face_landmark_5, transform_points, \
    warp_face_by_translation
from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.typing import Angle, BoundingBox, DownloadSet, FaceLandmark5, FaceLandmark68, ModelSet, \
    Prediction, Score, VisionFrame
from facefusion.workers.base_worker import BaseWorker


class FaceLandmarker(BaseWorker):
    MODEL_SET: ModelSet = \
        {
            '2dfan4':
                {
                    'hashes':
                        {
                            '2dfan4':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/2dfan4.hash',
                                    'path': resolve_relative_path('../.assets/models/2dfan4.hash')
                                }
                        },
                    'sources':
                        {
                            '2dfan4':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/2dfan4.onnx',
                                    'path': resolve_relative_path('../.assets/models/2dfan4.onnx')
                                }
                        },
                    'size': (256, 256)
                },
            'peppa_wutz':
                {
                    'hashes':
                        {
                            'peppa_wutz':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/peppa_wutz.hash',
                                    'path': resolve_relative_path('../.assets/models/peppa_wutz.hash')
                                }
                        },
                    'sources':
                        {
                            'peppa_wutz':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/peppa_wutz.onnx',
                                    'path': resolve_relative_path('../.assets/models/peppa_wutz.onnx')
                                }
                        },
                    'size': (256, 256)
                },
            'fan_68_5':
                {
                    'hashes':
                        {
                            'fan_68_5':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/fan_68_5.hash',
                                    'path': resolve_relative_path('../.assets/models/fan_68_5.hash')
                                }
                        },
                    'sources':
                        {
                            'fan_68_5':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/fan_68_5.onnx',
                                    'path': resolve_relative_path('../.assets/models/fan_68_5.onnx')
                                }
                        }
                }
        }

    default_model = 'many'
    model_key = 'face_landmarker_model'
    multi_model = True
    preload = True
    preferred_provider = 'cuda'

    def collect_model_downloads(self) -> Tuple[DownloadSet, DownloadSet]:
        model_hashes = \
            {
                'fan_68_5': self.MODEL_SET.get('fan_68_5').get('hashes').get('fan_68_5')
            }
        model_sources = \
            {
                'fan_68_5': self.MODEL_SET.get('fan_68_5').get('sources').get('fan_68_5')
            }

        if state_manager.get_item('face_landmarker_model') in ['many', '2dfan4']:
            model_hashes['2dfan4'] = self.MODEL_SET.get('2dfan4').get('hashes').get('2dfan4')
            model_sources['2dfan4'] = self.MODEL_SET.get('2dfan4').get('sources').get('2dfan4')
        if state_manager.get_item('face_landmarker_model') in ['many', 'peppa_wutz']:
            model_hashes['peppa_wutz'] = self.MODEL_SET.get('peppa_wutz').get('hashes').get('peppa_wutz')
            model_sources['peppa_wutz'] = self.MODEL_SET.get('peppa_wutz').get('sources').get('peppa_wutz')
        return model_hashes, model_sources

    def detect_face_landmarks(self, vision_frame: VisionFrame, bounding_box: BoundingBox, face_angle: Angle) -> Tuple[
        FaceLandmark68, Score]:
        face_landmark_2dfan4 = None
        face_landmark_peppa_wutz = None
        face_landmark_score_2dfan4 = 0.0
        face_landmark_score_peppa_wutz = 0.0

        if state_manager.get_item('face_landmarker_model') in ['many', '2dfan4']:
            face_landmark_2dfan4, face_landmark_score_2dfan4 = self.detect_with_2dfan4(vision_frame, bounding_box,
                                                                                       face_angle)
        if state_manager.get_item('face_landmarker_model') in ['many', 'peppa_wutz']:
            face_landmark_peppa_wutz, face_landmark_score_peppa_wutz = self.detect_with_peppa_wutz(vision_frame,
                                                                                                   bounding_box,
                                                                                                   face_angle)

        if face_landmark_score_2dfan4 > face_landmark_score_peppa_wutz - 0.2:
            return face_landmark_2dfan4, face_landmark_score_2dfan4
        return face_landmark_peppa_wutz, face_landmark_score_peppa_wutz

    def detect_with_2dfan4(self, temp_vision_frame: VisionFrame, bounding_box: BoundingBox, face_angle: Angle) -> Tuple[
        FaceLandmark68, Score]:
        model_size = self.MODEL_SET.get('2dfan4').get('size')
        scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)
        translation = (model_size[0] - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
        rotated_matrix, rotated_size = create_rotated_matrix_and_size(face_angle, model_size)
        crop_vision_frame, affine_matrix = warp_face_by_translation(temp_vision_frame, translation, scale, model_size)
        crop_vision_frame = cv2.warpAffine(crop_vision_frame, rotated_matrix, rotated_size)
        crop_vision_frame = self.conditional_optimize_contrast(crop_vision_frame)
        crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(numpy.float32) / 255.0
        face_landmark_68, face_heatmap = self.forward_with_2dfan4(crop_vision_frame)
        face_landmark_68 = face_landmark_68[:, :, :2][0] / 64 * 256
        face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(rotated_matrix))
        face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(affine_matrix))
        face_landmark_score_68 = numpy.amax(face_heatmap, axis=(2, 3))
        face_landmark_score_68 = numpy.mean(face_landmark_score_68)
        face_landmark_score_68 = numpy.interp(face_landmark_score_68, [0, 0.9], [0, 1])
        return face_landmark_68, face_landmark_score_68

    def detect_with_peppa_wutz(self, temp_vision_frame: VisionFrame, bounding_box: BoundingBox, face_angle: Angle) -> \
            Tuple[
                FaceLandmark68, Score]:
        model_size = self.MODEL_SET.get('peppa_wutz').get('size')
        scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)
        translation = (model_size[0] - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
        rotated_matrix, rotated_size = create_rotated_matrix_and_size(face_angle, model_size)
        crop_vision_frame, affine_matrix = warp_face_by_translation(temp_vision_frame, translation, scale, model_size)
        crop_vision_frame = cv2.warpAffine(crop_vision_frame, rotated_matrix, rotated_size)
        crop_vision_frame = self.conditional_optimize_contrast(crop_vision_frame)
        crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(numpy.float32) / 255.0
        crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0)
        prediction = self.forward_with_peppa_wutz(crop_vision_frame)
        face_landmark_68 = prediction.reshape(-1, 3)[:, :2] / 64 * model_size[0]
        face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(rotated_matrix))
        face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(affine_matrix))
        face_landmark_score_68 = prediction.reshape(-1, 3)[:, 2].mean()
        face_landmark_score_68 = numpy.interp(face_landmark_score_68, [0, 0.95], [0, 1])
        return face_landmark_68, face_landmark_score_68

    def conditional_optimize_contrast(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_RGB2Lab)
        if numpy.mean(crop_vision_frame[:, :, 0]) < 30:  # type:ignore[arg-type]
            crop_vision_frame[:, :, 0] = cv2.createCLAHE(clipLimit=2).apply(crop_vision_frame[:, :, 0])
        crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_Lab2RGB)
        return crop_vision_frame

    def estimate_face_landmark_68_5(self, face_landmark_5: FaceLandmark5) -> FaceLandmark68:
        affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, 'ffhq_512', (1, 1))
        face_landmark_5 = cv2.transform(face_landmark_5.reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
        face_landmark_68_5 = self.forward_fan_68_5(face_landmark_5)
        face_landmark_68_5 = cv2.transform(face_landmark_68_5.reshape(1, -1, 2),
                                           cv2.invertAffineTransform(affine_matrix)).reshape(-1, 2)
        return face_landmark_68_5

    def forward_with_2dfan4(self, crop_vision_frame: VisionFrame) -> Tuple[Prediction, Prediction]:
        face_landmarker = self.get_inference_pool().get('2dfan4')

        with conditional_thread_semaphore():
            prediction = face_landmarker.run(None,
                                             {
                                                 'input': [crop_vision_frame]
                                             })

        return prediction

    def forward_with_peppa_wutz(self, crop_vision_frame: VisionFrame) -> Prediction:
        face_landmarker = self.get_inference_pool().get('peppa_wutz')

        with conditional_thread_semaphore():
            prediction = face_landmarker.run(None,
                                             {
                                                 'input': crop_vision_frame
                                             })[0]

        return prediction

    def forward_fan_68_5(self, face_landmark_5: FaceLandmark5) -> FaceLandmark68:
        face_landmarker = self.get_inference_pool().get('fan_68_5')

        with conditional_thread_semaphore():
            face_landmark_68_5 = face_landmarker.run(None,
                                                     {
                                                         'input': [face_landmark_5]
                                                     })[0][0]

        return face_landmark_68_5
