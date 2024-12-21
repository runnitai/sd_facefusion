from argparse import ArgumentParser
from typing import List, Tuple

import cv2
import numpy

from facefusion import config, logger, process_manager, state_manager, wording
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
from facefusion.face_selector import find_similar_faces, sort_and_filter_faces
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.live_portrait import create_rotation, limit_expression
from facefusion.processors.typing import ExpressionRestorerInputs
from facefusion.processors.typing import LivePortraitExpression, LivePortraitFeatureVolume, LivePortraitMotionPoints, \
    LivePortraitPitch, LivePortraitRoll, LivePortraitScale, LivePortraitTranslation, LivePortraitYaw
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import conditional_thread_semaphore, thread_semaphore
from facefusion.typing import ApplyStateItem, Args, Face, QueuePayload, VisionFrame
from facefusion.vision import get_video_frame, read_image, read_static_image, write_image
from facefusion.workers.classes.face_masker import FaceMasker


def normalize_crop_frame(crop_vision_frame: VisionFrame) -> VisionFrame:
    crop_vision_frame = crop_vision_frame.transpose(1, 2, 0).clip(0, 1)
    crop_vision_frame = crop_vision_frame * 255.0
    crop_vision_frame = crop_vision_frame.astype(numpy.uint8)[:, :, ::-1]
    return crop_vision_frame


class ExpressionRestorer(BaseProcessor):
    """
    Processor for restoring expressions in faces within images or videos.
    """

    MODEL_SET = {
        'live_portrait': {
            'hashes': {
                'feature_extractor': {
                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_feature_extractor.hash',
                    'path': resolve_relative_path('../.assets/models/live_portrait_feature_extractor.hash'),
                },
                'motion_extractor': {
                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_motion_extractor.hash',
                    'path': resolve_relative_path('../.assets/models/live_portrait_motion_extractor.hash'),
                },
                'generator': {
                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_generator.hash',
                    'path': resolve_relative_path('../.assets/models/live_portrait_generator.hash'),
                },
            },
            'sources': {
                'feature_extractor': {
                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_feature_extractor.onnx',
                    'path': resolve_relative_path('../.assets/models/live_portrait_feature_extractor.onnx'),
                },
                'motion_extractor': {
                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_motion_extractor.onnx',
                    'path': resolve_relative_path('../.assets/models/live_portrait_motion_extractor.onnx'),
                },
                'generator': {
                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_generator.onnx',
                    'path': resolve_relative_path('../.assets/models/live_portrait_generator.onnx'),
                },
            },
            'template': 'arcface_128_v2',
            'size': (512, 512),
        }
    }

    model_key = "expression_restorer_model"
    priority = 3

    def register_args(self, program: ArgumentParser) -> None:
        group_processors = find_argument_group(program, "processors")
        if group_processors:
            group_processors.add_argument(
                "--expression-restorer-model",
                help=wording.get("help.expression_restorer_model"),
                default=config.get_str_value("processors.expression_restorer_model", "live_portrait"),
                choices=["live_portrait"],
            )
            group_processors.add_argument(
                "--expression-restorer-factor",
                help=wording.get("help.expression_restorer_factor"),
                type=int,
                default=config.get_int_value("processors.expression_restorer_factor", 80),
                choices=range(0, 101),
            )

    def apply_args(self, args: Args, apply_state_item: ApplyStateItem) -> None:
        apply_state_item("expression_restorer_model", args.get("expression_restorer_model"))
        apply_state_item("expression_restorer_factor", args.get("expression_restorer_factor"))

    def pre_process(self, mode: str) -> bool:
        if mode in ["output", "preview"]:
            target_path = state_manager.get_item("target_path")
            if not is_image(target_path) and not is_video(target_path):
                logger.error(wording.get("choose_image_or_video_target"), __name__)
                return False
        if mode == "output":
            output_path = state_manager.get_item("output_path")
            if not in_directory(output_path):
                logger.error(wording.get("specify_image_or_video_output"), __name__)
                return False
            if not same_file_extension([state_manager.get_item("target_path"), output_path]):
                logger.error(wording.get("match_target_and_output_extension"), __name__)
                return False
        return True

    def process_frame(self, inputs: ExpressionRestorerInputs) -> VisionFrame:
        reference_faces = inputs.get("reference_faces")
        reference_faces_2 = inputs.get("reference_faces_2")
        source_vision_frame = inputs.get("source_vision_frame")
        target_vision_frame = inputs.get("target_vision_frame")
        many_faces = sort_and_filter_faces(get_many_faces([target_vision_frame]))

        face_selector_mode = state_manager.get_item("face_selector_mode")
        if face_selector_mode == "many":
            if many_faces:
                for target_face in many_faces:
                    target_vision_frame = self.restore_expression(source_vision_frame, target_face, target_vision_frame)
        elif face_selector_mode == "one":
            target_face = get_one_face(many_faces)
            if target_face:
                target_vision_frame = self.restore_expression(source_vision_frame, target_face, target_vision_frame)
        elif face_selector_mode == "reference":
            for ref_faces in [reference_faces, reference_faces_2]:
                if ref_faces:
                    similar_faces = find_similar_faces(many_faces, ref_faces,
                                                       state_manager.get_item("reference_face_distance"))
                    for similar_face in similar_faces:
                        target_vision_frame = self.restore_expression(source_vision_frame, similar_face,
                                                                      target_vision_frame)
        return target_vision_frame

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        reference_faces, reference_faces_2 = get_reference_faces() if "reference" in state_manager.get_item(
            "face_selector_mode") else (None, None)
        output_frames = []
        for queue_payload in process_manager.manage(queue_payloads):
            frame_number = queue_payload.get("frame_number")
            if state_manager.get_item("trim_frame_start"):
                frame_number += state_manager.get_item("trim_frame_start")
            source_vision_frame = get_video_frame(state_manager.get_item("target_path"), frame_number)
            target_vision_path = queue_payload.get("frame_path")
            target_vision_frame = read_image(target_vision_path)
            output_vision_frame = self.process_frame(
                {
                    "reference_faces": reference_faces,
                    "reference_faces_2": reference_faces_2,
                    "source_vision_frame": source_vision_frame,
                    "target_vision_frame": target_vision_frame,
                }
            )
            write_image(target_vision_path, output_vision_frame)
            output_frames.append((frame_number, target_vision_path))
        return output_frames

    def process_image(self, target_path: str, output_path: str) -> None:
        reference_faces, reference_faces_2 = get_reference_faces() if "reference" in state_manager.get_item(
            "face_selector_mode") else (None, None)
        source_vision_frame = read_static_image(state_manager.get_item("target_path"))
        target_vision_frame = read_static_image(target_path)
        output_vision_frame = self.process_frame(
            {
                "reference_faces": reference_faces,
                "reference_faces_2": reference_faces_2,
                "source_vision_frame": source_vision_frame,
                "target_vision_frame": target_vision_frame,
            }
        )
        write_image(output_path, output_vision_frame)

    def restore_expression(self, source_vision_frame: VisionFrame, target_face: Face,
                           temp_vision_frame: VisionFrame) -> VisionFrame:
        masker = FaceMasker()
        model_template = self.get_model_options().get("template")
        model_size = self.get_model_options().get("size")
        expression_restorer_factor = float(
            numpy.interp(
                float(state_manager.get_item("expression_restorer_factor")), [0, 100], [0, 1.2]
            )
        )
        source_vision_frame = cv2.resize(source_vision_frame, temp_vision_frame.shape[:2][::-1])
        source_crop_vision_frame, _ = warp_face_by_face_landmark_5(
            source_vision_frame, target_face.landmark_set.get("5/68"), model_template, model_size
        )
        target_crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
            temp_vision_frame, target_face.landmark_set.get("5/68"), model_template, model_size
        )
        box_mask = masker.create_static_box_mask(
            target_crop_vision_frame.shape[:2][::-1], state_manager.get_item("face_mask_blur"), (0, 0, 0, 0)
        )
        crop_masks = [box_mask]

        if "occlusion" in state_manager.get_item("face_mask_types"):
            occlusion_mask = masker.create_occlusion_mask(target_crop_vision_frame)
            crop_masks.append(occlusion_mask)

        source_crop_vision_frame = self.prepare_crop_frame(source_crop_vision_frame)
        target_crop_vision_frame = self.prepare_crop_frame(target_crop_vision_frame)
        target_crop_vision_frame = self.apply_restore(
            source_crop_vision_frame, target_crop_vision_frame, expression_restorer_factor
        )
        target_crop_vision_frame = normalize_crop_frame(target_crop_vision_frame)
        crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
        temp_vision_frame = paste_back(temp_vision_frame, target_crop_vision_frame, crop_mask, affine_matrix)
        return temp_vision_frame

    def apply_restore(self, source_crop_vision_frame: VisionFrame, target_crop_vision_frame: VisionFrame,
                      expression_restorer_factor: float) -> VisionFrame:
        feature_volume = self.forward_extract_feature(target_crop_vision_frame)
        source_expression = self.forward_extract_motion(source_crop_vision_frame)[5]
        pitch, yaw, roll, scale, translation, target_expression, motion_points = self.forward_extract_motion(
            target_crop_vision_frame
        )
        rotation = create_rotation(pitch, yaw, roll)
        source_expression[:, [0, 4, 5, 8, 9]] = target_expression[:, [0, 4, 5, 8, 9]]
        source_expression = source_expression * expression_restorer_factor + target_expression * (
                1 - expression_restorer_factor
        )
        source_expression = limit_expression(source_expression)
        source_motion_points = scale * (motion_points @ rotation.T + source_expression) + translation
        target_motion_points = scale * (motion_points @ rotation.T + target_expression) + translation
        crop_vision_frame = self.forward_generate_frame(feature_volume, source_motion_points, target_motion_points)
        return crop_vision_frame

    def forward_extract_feature(self, crop_vision_frame: VisionFrame) -> LivePortraitFeatureVolume:
        feature_extractor = self.get_inference_pool().get("feature_extractor")

        with conditional_thread_semaphore():
            feature_volume = feature_extractor.run(None, {"input": crop_vision_frame})[0]

        return feature_volume

    def forward_extract_motion(self, crop_vision_frame: VisionFrame) -> Tuple[
        LivePortraitPitch,
        LivePortraitYaw,
        LivePortraitRoll,
        LivePortraitScale,
        LivePortraitTranslation,
        LivePortraitExpression,
        LivePortraitMotionPoints,
    ]:
        motion_extractor = self.get_inference_pool().get("motion_extractor")

        with conditional_thread_semaphore():
            pitch, yaw, roll, scale, translation, expression, motion_points = motion_extractor.run(
                None, {"input": crop_vision_frame}
            )

        return pitch, yaw, roll, scale, translation, expression, motion_points

    def forward_generate_frame(
            self, feature_volume: LivePortraitFeatureVolume, source_motion_points: LivePortraitMotionPoints,
            target_motion_points: LivePortraitMotionPoints
    ) -> VisionFrame:
        generator = self.get_inference_pool().get("generator")

        with thread_semaphore():
            crop_vision_frame = generator.run(
                None,
                {
                    "feature_volume": feature_volume,
                    "source": source_motion_points,
                    "target": target_motion_points,
                },
            )[0][0]

        return crop_vision_frame

    def prepare_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        model_size = self.get_model_options().get("size")
        prepare_size = (model_size[0] // 2, model_size[1] // 2)
        crop_vision_frame = cv2.resize(crop_vision_frame, prepare_size, interpolation=cv2.INTER_AREA)
        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        crop_vision_frame = numpy.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis=0).astype(numpy.float32)
        return crop_vision_frame
