from argparse import ArgumentParser
from typing import List, Tuple, Any

import cv2
import numpy

from facefusion import logger, state_manager, wording, config, process_manager
from facefusion.common_helper import create_float_metavar
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.face_helper import paste_back, scale_face_landmark_5, warp_face_by_face_landmark_5
from facefusion.face_selector import find_similar_faces, sort_and_filter_faces
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import resolve_relative_path, is_image, is_video, in_directory, same_file_extension
from facefusion.jobs import job_store
from facefusion.processors import choices as processors_choices
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.live_portrait import create_rotation, limit_euler_angles, limit_expression
from facefusion.processors.typing import (
    FaceEditorInputs, LivePortraitExpression, LivePortraitFeatureVolume,
    LivePortraitMotionPoints, LivePortraitPitch, LivePortraitRoll,
    LivePortraitRotation, LivePortraitScale, LivePortraitTranslation,
    LivePortraitYaw
)
from facefusion.thread_helper import conditional_thread_semaphore, thread_semaphore
from facefusion.typing import ModelSet, Face, FaceLandmark68, VisionFrame, ProcessMode, QueuePayload
from facefusion.vision import read_image, read_static_image, write_image
from facefusion.workers.classes.face_masker import FaceMasker


def normalize_crop_frame(crop_vision_frame: VisionFrame) -> VisionFrame:
    crop_vision_frame = crop_vision_frame.transpose(1, 2, 0).clip(0, 1)
    crop_vision_frame = (crop_vision_frame * 255.0)
    crop_vision_frame = crop_vision_frame.astype(numpy.uint8)[:, :, ::-1]
    return crop_vision_frame


def calc_distance_ratio(face_landmark_68: FaceLandmark68, top_index: int, bottom_index: int, left_index: int,
                        right_index: int) -> float:
    vertical_direction = face_landmark_68[top_index] - face_landmark_68[bottom_index]
    horizontal_direction = face_landmark_68[left_index] - face_landmark_68[right_index]
    distance_ratio = float(numpy.linalg.norm(vertical_direction) / (numpy.linalg.norm(horizontal_direction) + 1e-6))
    return distance_ratio


def edit_head_rotation(pitch: LivePortraitPitch, yaw: LivePortraitYaw,
                       roll: LivePortraitRoll) -> LivePortraitRotation:
    face_editor_head_pitch = state_manager.get_item('face_editor_head_pitch') or 0
    face_editor_head_yaw = state_manager.get_item('face_editor_head_yaw') or 0
    face_editor_head_roll = state_manager.get_item('face_editor_head_roll') or 0

    edit_pitch = pitch + float(numpy.interp(face_editor_head_pitch, [-1, 1], [20, -20]))
    edit_yaw = yaw + float(numpy.interp(face_editor_head_yaw, [-1, 1], [60, -60]))
    edit_roll = roll + float(numpy.interp(face_editor_head_roll, [-1, 1], [-15, 15]))
    edit_pitch, edit_yaw, edit_roll = limit_euler_angles(pitch, yaw, roll, edit_pitch, edit_yaw, edit_roll)
    rotation = create_rotation(edit_pitch, edit_yaw, edit_roll)
    return rotation


def edit_mouth_smile(expression: LivePortraitExpression) -> LivePortraitExpression:
    face_editor_mouth_smile = state_manager.get_item('face_editor_mouth_smile') or 0

    if face_editor_mouth_smile > 0:
        expression[0, 20, 1] -= numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.015, 0.015])
        expression[0, 14, 1] -= numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.025, 0.025])
        expression[0, 17, 1] += numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.01, 0.01])
        expression[0, 17, 2] += numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.004, 0.004])
        expression[0, 3, 1] -= numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.0045, 0.0045])
        expression[0, 7, 1] -= numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.0045, 0.0045])
    else:
        expression[0, 14, 1] -= numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.02, 0.02])
        expression[0, 17, 1] += numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.003, 0.003])
        expression[0, 19, 1] += numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.02, 0.02])
        expression[0, 19, 2] -= numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.005, 0.005])
        expression[0, 20, 2] += numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.01, 0.01])
        expression[0, 3, 1] += numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.0045, 0.0045])
        expression[0, 7, 1] += numpy.interp(face_editor_mouth_smile, [-1, 1], [-0.0045, 0.0045])
    return expression


def edit_mouth_purse(expression: LivePortraitExpression) -> LivePortraitExpression:
    face_editor_mouth_purse = state_manager.get_item('face_editor_mouth_purse') or 0

    if face_editor_mouth_purse > 0:
        expression[0, 19, 1] -= numpy.interp(face_editor_mouth_purse, [-1, 1], [-0.04, 0.04])
        expression[0, 19, 2] -= numpy.interp(face_editor_mouth_purse, [-1, 1], [-0.02, 0.02])
    else:
        expression[0, 14, 1] -= numpy.interp(face_editor_mouth_purse, [-1, 1], [-0.02, 0.02])
        expression[0, 17, 2] += numpy.interp(face_editor_mouth_purse, [-1, 1], [-0.01, 0.01])
        expression[0, 19, 2] -= numpy.interp(face_editor_mouth_purse, [-1, 1], [-0.015, 0.015])
        expression[0, 20, 2] -= numpy.interp(face_editor_mouth_purse, [-1, 1], [-0.002, 0.002])
    return expression


def edit_mouth_pout(expression: LivePortraitExpression) -> LivePortraitExpression:
    face_editor_mouth_pout = state_manager.get_item('face_editor_mouth_pout') or 0

    if face_editor_mouth_pout > 0:
        expression[0, 19, 1] -= numpy.interp(face_editor_mouth_pout, [-1, 1], [-0.022, 0.022])
        expression[0, 19, 2] += numpy.interp(face_editor_mouth_pout, [-1, 1], [-0.025, 0.025])
        expression[0, 20, 2] -= numpy.interp(face_editor_mouth_pout, [-1, 1], [-0.002, 0.002])
    else:
        expression[0, 19, 1] += numpy.interp(face_editor_mouth_pout, [-1, 1], [-0.022, 0.022])
        expression[0, 19, 2] += numpy.interp(face_editor_mouth_pout, [-1, 1], [-0.025, 0.025])
        expression[0, 20, 2] -= numpy.interp(face_editor_mouth_pout, [-1, 1], [-0.002, 0.002])
    return expression


def edit_mouth_position(expression: LivePortraitExpression) -> LivePortraitExpression:
    face_editor_mouth_position_horizontal = state_manager.get_item('face_editor_mouth_position_horizontal') or 0
    face_editor_mouth_position_vertical = state_manager.get_item('face_editor_mouth_position_vertical') or 0

    expression[0, 19, 0] += numpy.interp(face_editor_mouth_position_horizontal, [-1, 1], [-0.05, 0.05])
    expression[0, 20, 0] += numpy.interp(face_editor_mouth_position_horizontal, [-1, 1], [-0.04, 0.04])
    if face_editor_mouth_position_vertical > 0:
        expression[0, 19, 1] -= numpy.interp(face_editor_mouth_position_vertical, [-1, 1], [-0.04, 0.04])
        expression[0, 20, 1] -= numpy.interp(face_editor_mouth_position_vertical, [-1, 1], [-0.02, 0.02])
    else:
        expression[0, 19, 1] -= numpy.interp(face_editor_mouth_position_vertical, [-1, 1], [-0.05, 0.05])
        expression[0, 20, 1] -= numpy.interp(face_editor_mouth_position_vertical, [-1, 1], [-0.04, 0.04])
    return expression


def edit_mouth_grim(expression: LivePortraitExpression) -> LivePortraitExpression:
    face_editor_mouth_grim = state_manager.get_item('face_editor_mouth_grim') or 0
    if face_editor_mouth_grim > 0:
        expression[0, 17, 2] -= numpy.interp(face_editor_mouth_grim, [-1, 1], [-0.005, 0.005])
        expression[0, 19, 2] += numpy.interp(face_editor_mouth_grim, [-1, 1], [-0.01, 0.01])
        expression[0, 20, 1] -= numpy.interp(face_editor_mouth_grim, [-1, 1], [-0.06, 0.06])
        expression[0, 20, 2] -= numpy.interp(face_editor_mouth_grim, [-1, 1], [-0.03, 0.03])
    else:
        expression[0, 19, 1] -= numpy.interp(face_editor_mouth_grim, [-1, 1], [-0.05, 0.05])
        expression[0, 19, 2] -= numpy.interp(face_editor_mouth_grim, [-1, 1], [-0.02, 0.02])
        expression[0, 20, 2] -= numpy.interp(face_editor_mouth_grim, [-1, 1], [-0.03, 0.03])
    return expression


def edit_eye_gaze(expression: LivePortraitExpression) -> LivePortraitExpression:
    face_editor_eye_gaze_horizontal = state_manager.get_item('face_editor_eye_gaze_horizontal') or 0
    face_editor_eye_gaze_vertical = state_manager.get_item('face_editor_eye_gaze_vertical') or 0

    if face_editor_eye_gaze_horizontal > 0:
        expression[0, 11, 0] += numpy.interp(face_editor_eye_gaze_horizontal, [-1, 1], [-0.015, 0.015])
        expression[0, 15, 0] += numpy.interp(face_editor_eye_gaze_horizontal, [-1, 1], [-0.020, 0.020])
    else:
        expression[0, 11, 0] += numpy.interp(face_editor_eye_gaze_horizontal, [-1, 1], [-0.020, 0.020])
        expression[0, 15, 0] += numpy.interp(face_editor_eye_gaze_horizontal, [-1, 1], [-0.015, 0.015])
    expression[0, 1, 1] += numpy.interp(face_editor_eye_gaze_vertical, [-1, 1], [-0.0025, 0.0025])
    expression[0, 2, 1] -= numpy.interp(face_editor_eye_gaze_vertical, [-1, 1], [-0.0025, 0.0025])
    expression[0, 11, 1] -= numpy.interp(face_editor_eye_gaze_vertical, [-1, 1], [-0.010, 0.010])
    expression[0, 13, 1] -= numpy.interp(face_editor_eye_gaze_vertical, [-1, 1], [-0.005, 0.005])
    expression[0, 15, 1] -= numpy.interp(face_editor_eye_gaze_vertical, [-1, 1], [-0.010, 0.010])
    expression[0, 16, 1] -= numpy.interp(face_editor_eye_gaze_vertical, [-1, 1], [-0.005, 0.005])
    return expression


def edit_eyebrow_direction(expression: LivePortraitExpression) -> LivePortraitExpression:
    face_editor_eyebrow = state_manager.get_item('face_editor_eyebrow_direction') or 0

    if face_editor_eyebrow > 0:
        expression[0, 1, 1] += numpy.interp(face_editor_eyebrow, [-1, 1], [-0.015, 0.015])
        expression[0, 2, 1] -= numpy.interp(face_editor_eyebrow, [-1, 1], [-0.020, 0.020])
    else:
        expression[0, 1, 0] -= numpy.interp(face_editor_eyebrow, [-1, 1], [-0.015, 0.015])
        expression[0, 2, 0] += numpy.interp(face_editor_eyebrow, [-1, 1], [-0.020, 0.020])
        expression[0, 1, 1] += numpy.interp(face_editor_eyebrow, [-1, 1], [-0.005, 0.005])
        expression[0, 2, 1] -= numpy.interp(face_editor_eyebrow, [-1, 1], [-0.005, 0.005])
    return expression


class FaceEditor(BaseProcessor):
    MODEL_SET: ModelSet = \
        {
            'live_portrait':
                {
                    'hashes':
                        {
                            'feature_extractor':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_feature_extractor.hash',
                                    'path': resolve_relative_path(
                                        '../.assets/models/live_portrait_feature_extractor.hash')
                                },
                            'motion_extractor':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_motion_extractor.hash',
                                    'path': resolve_relative_path(
                                        '../.assets/models/live_portrait_motion_extractor.hash')
                                },
                            'eye_retargeter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_eye_retargeter.hash',
                                    'path': resolve_relative_path('../.assets/models/live_portrait_eye_retargeter.hash')
                                },
                            'lip_retargeter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_lip_retargeter.hash',
                                    'path': resolve_relative_path('../.assets/models/live_portrait_lip_retargeter.hash')
                                },
                            'stitcher':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_stitcher.hash',
                                    'path': resolve_relative_path('../.assets/models/live_portrait_stitcher.hash')
                                },
                            'generator':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_generator.hash',
                                    'path': resolve_relative_path('../.assets/models/live_portrait_generator.hash')
                                }
                        },
                    'sources':
                        {
                            'feature_extractor':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_feature_extractor.onnx',
                                    'path': resolve_relative_path(
                                        '../.assets/models/live_portrait_feature_extractor.onnx')
                                },
                            'motion_extractor':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_motion_extractor.onnx',
                                    'path': resolve_relative_path(
                                        '../.assets/models/live_portrait_motion_extractor.onnx')
                                },
                            'eye_retargeter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_eye_retargeter.onnx',
                                    'path': resolve_relative_path('../.assets/models/live_portrait_eye_retargeter.onnx')
                                },
                            'lip_retargeter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_lip_retargeter.onnx',
                                    'path': resolve_relative_path('../.assets/models/live_portrait_lip_retargeter.onnx')
                                },
                            'stitcher':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_stitcher.onnx',
                                    'path': resolve_relative_path('../.assets/models/live_portrait_stitcher.onnx')
                                },
                            'generator':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_generator.onnx',
                                    'path': resolve_relative_path('../.assets/models/live_portrait_generator.onnx')
                                }
                        },
                    'template': 'ffhq_512',
                    'size': (512, 512)
                }
        }

    default_model = 'live_portrait'
    model_key = 'face_editor_model'
    priority = 9

    def __init__(self):
        super().__init__()
        self.model_key = 'face_editor_model'

    def register_args(self, program: ArgumentParser) -> None:
        group_processors = program.add_argument_group('processors')
        group_processors.add_argument('--face-editor-model',
                                      help=wording.get('help.face_editor_model'),
                                      default=config.get_str_value('processors.face_editor_model', 'live_portrait'),
                                      choices=self.list_models())
        group_processors.add_argument('--face-editor-eyebrow-direction',
                                      help=wording.get('help.face_editor_eyebrow_direction'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_eyebrow_direction', '0'),
                                      choices=processors_choices.face_editor_eyebrow_direction_range,
                                      metavar=create_float_metavar(
                                          processors_choices.face_editor_eyebrow_direction_range))
        group_processors.add_argument('--face-editor-eye-gaze-horizontal',
                                      help=wording.get('help.face_editor_eye_gaze_horizontal'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_eye_gaze_horizontal', '0'),
                                      choices=processors_choices.face_editor_eye_gaze_horizontal_range,
                                      metavar=create_float_metavar(
                                          processors_choices.face_editor_eye_gaze_horizontal_range))
        group_processors.add_argument('--face-editor-eye-gaze-vertical',
                                      help=wording.get('help.face_editor_eye_gaze_vertical'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_eye_gaze_vertical', '0'),
                                      choices=processors_choices.face_editor_eye_gaze_vertical_range,
                                      metavar=create_float_metavar(
                                          processors_choices.face_editor_eye_gaze_vertical_range))
        group_processors.add_argument('--face-editor-eye-open-ratio',
                                      help=wording.get('help.face_editor_eye_open_ratio'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_eye_open_ratio', '0'),
                                      choices=processors_choices.face_editor_eye_open_ratio_range,
                                      metavar=create_float_metavar(processors_choices.face_editor_eye_open_ratio_range))
        group_processors.add_argument('--face-editor-lip-open-ratio',
                                      help=wording.get('help.face_editor_lip_open_ratio'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_lip_open_ratio', '0'),
                                      choices=processors_choices.face_editor_lip_open_ratio_range,
                                      metavar=create_float_metavar(processors_choices.face_editor_lip_open_ratio_range))
        group_processors.add_argument('--face-editor-mouth-grim',
                                      help=wording.get('help.face_editor_mouth_grim'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_mouth_grim', '0'),
                                      choices=processors_choices.face_editor_mouth_grim_range,
                                      metavar=create_float_metavar(processors_choices.face_editor_mouth_grim_range))
        group_processors.add_argument('--face-editor-mouth-pout',
                                      help=wording.get('help.face_editor_mouth_pout'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_mouth_pout', '0'),
                                      choices=processors_choices.face_editor_mouth_pout_range,
                                      metavar=create_float_metavar(processors_choices.face_editor_mouth_pout_range))
        group_processors.add_argument('--face-editor-mouth-purse',
                                      help=wording.get('help.face_editor_mouth_purse'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_mouth_purse', '0'),
                                      choices=processors_choices.face_editor_mouth_purse_range,
                                      metavar=create_float_metavar(processors_choices.face_editor_mouth_purse_range))
        group_processors.add_argument('--face-editor-mouth-smile',
                                      help=wording.get('help.face_editor_mouth_smile'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_mouth_smile', '0'),
                                      choices=processors_choices.face_editor_mouth_smile_range,
                                      metavar=create_float_metavar(processors_choices.face_editor_mouth_smile_range))
        group_processors.add_argument('--face-editor-mouth-position-horizontal',
                                      help=wording.get('help.face_editor_mouth_position_horizontal'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_mouth_position_horizontal',
                                                                     '0'),
                                      choices=processors_choices.face_editor_mouth_position_horizontal_range,
                                      metavar=create_float_metavar(
                                          processors_choices.face_editor_mouth_position_horizontal_range))
        group_processors.add_argument('--face-editor-mouth-position-vertical',
                                      help=wording.get('help.face_editor_mouth_position_vertical'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_mouth_position_vertical',
                                                                     '0'),
                                      choices=processors_choices.face_editor_mouth_position_vertical_range,
                                      metavar=create_float_metavar(
                                          processors_choices.face_editor_mouth_position_vertical_range))
        group_processors.add_argument('--face-editor-head-pitch',
                                      help=wording.get('help.face_editor_head_pitch'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_head_pitch', '0'),
                                      choices=processors_choices.face_editor_head_pitch_range,
                                      metavar=create_float_metavar(processors_choices.face_editor_head_pitch_range))
        group_processors.add_argument('--face-editor-head-yaw',
                                      help=wording.get('help.face_editor_head_yaw'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_head_yaw', '0'),
                                      choices=processors_choices.face_editor_head_yaw_range,
                                      metavar=create_float_metavar(processors_choices.face_editor_head_yaw_range))
        group_processors.add_argument('--face-editor-head-roll',
                                      help=wording.get('help.face_editor_head_roll'),
                                      type=float,
                                      default=config.get_float_value('processors.face_editor_head_roll', '0'),
                                      choices=processors_choices.face_editor_head_roll_range,
                                      metavar=create_float_metavar(processors_choices.face_editor_head_roll_range))
        job_store.register_step_keys([
            'face_editor_model',
            'face_editor_eyebrow_direction',
            'face_editor_eye_gaze_horizontal',
            'face_editor_eye_gaze_vertical',
            'face_editor_eye_open_ratio',
            'face_editor_lip_open_ratio',
            'face_editor_mouth_grim',
            'face_editor_mouth_pout',
            'face_editor_mouth_purse',
            'face_editor_mouth_smile',
            'face_editor_mouth_position_horizontal',
            'face_editor_mouth_position_vertical',
            'face_editor_head_pitch',
            'face_editor_head_yaw',
            'face_editor_head_roll'
        ])

    def apply_args(self, args: Any, apply_state_item: Any) -> None:
        apply_state_item('face_editor_model', args.get('face_editor_model'))
        apply_state_item('face_editor_eyebrow_direction', args.get('face_editor_eyebrow_direction'))
        apply_state_item('face_editor_eye_gaze_horizontal', args.get('face_editor_eye_gaze_horizontal'))
        apply_state_item('face_editor_eye_gaze_vertical', args.get('face_editor_eye_gaze_vertical'))
        apply_state_item('face_editor_eye_open_ratio', args.get('face_editor_eye_open_ratio'))
        apply_state_item('face_editor_lip_open_ratio', args.get('face_editor_lip_open_ratio'))
        apply_state_item('face_editor_mouth_grim', args.get('face_editor_mouth_grim'))
        apply_state_item('face_editor_mouth_pout', args.get('face_editor_mouth_pout'))
        apply_state_item('face_editor_mouth_purse', args.get('face_editor_mouth_purse'))
        apply_state_item('face_editor_mouth_smile', args.get('face_editor_mouth_smile'))
        apply_state_item('face_editor_mouth_position_horizontal', args.get('face_editor_mouth_position_horizontal'))
        apply_state_item('face_editor_mouth_position_vertical', args.get('face_editor_mouth_position_vertical'))
        apply_state_item('face_editor_head_pitch', args.get('face_editor_head_pitch'))
        apply_state_item('face_editor_head_yaw', args.get('face_editor_head_yaw'))
        apply_state_item('face_editor_head_roll', args.get('face_editor_head_roll'))

    def pre_process(self, mode: ProcessMode) -> bool:
        if mode in ['output', 'preview'] and not is_image(state_manager.get_item('target_path')) and not is_video(
                state_manager.get_item('target_path')):
            logger.error(wording.get('choose_image_or_video_target') + wording.get('exclamation_mark'), __name__)
            return False
        if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
            logger.error(wording.get('specify_image_or_video_output') + wording.get('exclamation_mark'), __name__)
            return False
        if mode == 'output' and not same_file_extension(
                [state_manager.get_item('target_path'), state_manager.get_item('output_path')]):
            logger.error(wording.get('match_target_and_output_extension') + wording.get('exclamation_mark'), __name__)
            return False
        self.inference_pool = self.get_inference_pool()
        return True

    def process_frame(self, inputs: FaceEditorInputs) -> VisionFrame:
        reference_faces = inputs.get('reference_faces')
        target_vision_frame = inputs.get('target_vision_frame')
        many_faces = sort_and_filter_faces(get_many_faces([target_vision_frame]))

        if state_manager.get_item('face_selector_mode') == 'many':
            if many_faces:
                for target_face in many_faces:
                    target_vision_frame = self.edit_face(target_face, target_vision_frame)
        if state_manager.get_item('face_selector_mode') == 'one':
            target_face = get_one_face(many_faces)
            if target_face:
                target_vision_frame = self.edit_face(target_face, target_vision_frame)
        if state_manager.get_item('face_selector_mode') == 'reference':
            for src_face_idx, ref_faces in reference_faces.items():
                similar_faces = find_similar_faces(many_faces, ref_faces,
                                                   state_manager.get_item('reference_face_distance'))
                if similar_faces:
                    for similar_face in similar_faces:
                        target_vision_frame = self.edit_face(similar_face, target_vision_frame)
        return target_vision_frame

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        output_frames = []
        for queue_payload in process_manager.manage(queue_payloads):
            target_vision_path = queue_payload['frame_path']
            target_frame_number = queue_payload['frame_number']
            reference_faces = queue_payload['reference_faces']
            target_vision_frame = read_image(target_vision_path)
            result_frame = self.process_frame(
                {
                    'reference_faces': reference_faces,
                    'target_vision_frame': target_vision_frame,
                    'target_frame_number': target_frame_number
                })
            write_image(target_vision_path, result_frame)
            output_frames.append((target_frame_number, target_vision_path))
        return output_frames

    def process_image(self, target_path: str, output_path: str, reference_faces=None) -> None:
        if reference_faces is None:
            reference_faces = (
                get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else (None, None))
        target_vision_frame = read_static_image(target_path)
        output_vision_frame = self.process_frame(
            {
                'reference_faces': reference_faces,
                'target_vision_frame': target_vision_frame
            })
        write_image(output_path, output_vision_frame)

    def edit_face(self, target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
        masker = FaceMasker()
        model_template = self.get_model_options().get('template')
        model_size = self.get_model_options().get('size')
        face_landmark_5 = scale_face_landmark_5(target_face.landmark_set.get('5/68'), 1.5)
        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5,
                                                                        model_template,
                                                                        model_size)
        box_mask = masker.create_static_box_mask(crop_vision_frame.shape[:2][::-1],
                                                 state_manager.get_item('face_mask_blur'),
                                                 (0, 0, 0, 0))
        crop_vision_frame = self.prepare_crop_frame(crop_vision_frame)
        crop_vision_frame = self.apply_edit(crop_vision_frame, target_face.landmark_set.get('68'))
        crop_vision_frame = normalize_crop_frame(crop_vision_frame)
        temp_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, box_mask, affine_matrix)
        return temp_vision_frame

    def apply_edit(self, crop_vision_frame: VisionFrame, face_landmark_68: FaceLandmark68) -> VisionFrame:
        feature_volume = self.forward_extract_feature(crop_vision_frame)
        pitch, yaw, roll, scale, translation, expression, motion_points = self.forward_extract_motion(crop_vision_frame)
        rotation = create_rotation(pitch, yaw, roll)
        motion_points_target = scale * (motion_points @ rotation.T + expression) + translation
        expression = edit_eye_gaze(expression)
        expression = edit_mouth_grim(expression)
        expression = edit_mouth_position(expression)
        expression = edit_mouth_pout(expression)
        expression = edit_mouth_purse(expression)
        expression = edit_mouth_smile(expression)
        expression = edit_eyebrow_direction(expression)
        expression = limit_expression(expression)
        rotation = edit_head_rotation(pitch, yaw, roll)
        motion_points_source = motion_points @ rotation.T
        motion_points_source += expression
        motion_points_source *= scale
        motion_points_source += translation
        motion_points_source += self.edit_eye_open(motion_points_target, face_landmark_68)
        motion_points_source += self.edit_lip_open(motion_points_target, face_landmark_68)
        motion_points_source = self.forward_stitch_motion_points(motion_points_source, motion_points_target)
        crop_vision_frame = self.forward_generate_frame(feature_volume, motion_points_source, motion_points_target)
        return crop_vision_frame

    def forward_extract_feature(self, crop_vision_frame: VisionFrame) -> LivePortraitFeatureVolume:
        feature_extractor = self.inference_pool.get('feature_extractor')

        with conditional_thread_semaphore():
            feature_volume = feature_extractor.run(None,
                                                   {
                                                       'input': crop_vision_frame
                                                   })[0]

        return feature_volume

    def forward_extract_motion(self, crop_vision_frame: VisionFrame) -> Tuple[
        LivePortraitPitch, LivePortraitYaw, LivePortraitRoll, LivePortraitScale, LivePortraitTranslation, LivePortraitExpression, LivePortraitMotionPoints]:
        motion_extractor = self.inference_pool.get('motion_extractor')

        with conditional_thread_semaphore():
            pitch, yaw, roll, scale, translation, expression, motion_points = motion_extractor.run(None,
                                                                                                   {
                                                                                                       'input': crop_vision_frame
                                                                                                   })

        return pitch, yaw, roll, scale, translation, expression, motion_points

    def forward_retarget_eye(self, eye_motion_points: LivePortraitMotionPoints) -> LivePortraitMotionPoints:
        eye_retargeter = self.inference_pool.get('eye_retargeter')

        with conditional_thread_semaphore():
            eye_motion_points = eye_retargeter.run(None,
                                                   {
                                                       'input': eye_motion_points
                                                   })[0]

        return eye_motion_points

    def forward_retarget_lip(self, lip_motion_points: LivePortraitMotionPoints) -> LivePortraitMotionPoints:
        lip_retargeter = self.inference_pool.get('lip_retargeter')

        with conditional_thread_semaphore():
            lip_motion_points = lip_retargeter.run(None,
                                                   {
                                                       'input': lip_motion_points
                                                   })[0]

        return lip_motion_points

    def forward_stitch_motion_points(self, source_motion_points: LivePortraitMotionPoints,
                                     target_motion_points: LivePortraitMotionPoints) -> LivePortraitMotionPoints:
        stitcher = self.inference_pool.get('stitcher')

        with thread_semaphore():
            motion_points = stitcher.run(None,
                                         {
                                             'source': source_motion_points,
                                             'target': target_motion_points
                                         })[0]

        return motion_points

    def forward_generate_frame(self, feature_volume: LivePortraitFeatureVolume,
                               source_motion_points: LivePortraitMotionPoints,
                               target_motion_points: LivePortraitMotionPoints) -> VisionFrame:
        generator = self.inference_pool.get('generator')

        with thread_semaphore():
            crop_vision_frame = generator.run(None,
                                              {
                                                  'feature_volume': feature_volume,
                                                  'source': source_motion_points,
                                                  'target': target_motion_points
                                              })[0][0]

        return crop_vision_frame

    def edit_eye_open(self, motion_points: LivePortraitMotionPoints,
                      face_landmark_68: FaceLandmark68) -> LivePortraitMotionPoints:
        face_editor_eye_open_ratio = state_manager.get_item('face_editor_eye_open_ratio') or 0
        left_eye_ratio = calc_distance_ratio(face_landmark_68, 37, 40, 39, 36)
        right_eye_ratio = calc_distance_ratio(face_landmark_68, 43, 46, 45, 42)

        if face_editor_eye_open_ratio < 0:
            eye_motion_points = numpy.concatenate([motion_points.ravel(), [left_eye_ratio, right_eye_ratio, 0.0]])
        else:
            eye_motion_points = numpy.concatenate([motion_points.ravel(), [left_eye_ratio, right_eye_ratio, 0.6]])
        eye_motion_points = eye_motion_points.reshape(1, -1).astype(numpy.float32)
        eye_motion_points = self.forward_retarget_eye(eye_motion_points) * numpy.abs(face_editor_eye_open_ratio)
        eye_motion_points = eye_motion_points.reshape(-1, 21, 3)
        return eye_motion_points

    def edit_lip_open(self, motion_points: LivePortraitMotionPoints,
                      face_landmark_68: FaceLandmark68) -> LivePortraitMotionPoints:
        face_editor_lip_open_ratio = state_manager.get_item('face_editor_lip_open_ratio') or 0
        lip_ratio = calc_distance_ratio(face_landmark_68, 62, 66, 54, 48)

        if face_editor_lip_open_ratio < 0:
            lip_motion_points = numpy.concatenate([motion_points.ravel(), [lip_ratio, 0.0]])
        else:
            lip_motion_points = numpy.concatenate([motion_points.ravel(), [lip_ratio, 1.0]])
        lip_motion_points = lip_motion_points.reshape(1, -1).astype(numpy.float32)
        lip_motion_points = self.forward_retarget_lip(lip_motion_points) * numpy.abs(face_editor_lip_open_ratio)
        lip_motion_points = lip_motion_points.reshape(-1, 21, 3)
        return lip_motion_points

    def prepare_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        model_size = self.get_model_options().get('size')
        prepare_size = (model_size[0] // 2, model_size[1] // 2)
        crop_vision_frame = cv2.resize(crop_vision_frame, prepare_size, interpolation=cv2.INTER_AREA)
        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        crop_vision_frame = numpy.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis=0).astype(numpy.float32)
        return crop_vision_frame
