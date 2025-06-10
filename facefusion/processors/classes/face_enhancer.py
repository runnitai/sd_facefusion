from argparse import ArgumentParser
from typing import List, Tuple

import cv2
import numpy

from facefusion import config, logger, state_manager, wording, process_manager
from facefusion.common_helper import create_int_metavar
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
from facefusion.face_selector import find_similar_faces, sort_and_filter_faces
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.jobs import job_store
from facefusion.processors import choices as processors_choices
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.typing import FaceEnhancerInputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import thread_semaphore
from facefusion.typing import ApplyStateItem, Args, Face, ModelSet, ProcessMode, \
    QueuePayload, VisionFrame
from facefusion.vision import read_image, read_static_image, write_image
from facefusion.workers.classes.face_masker import FaceMasker
from facefusion.workers.core import clear_worker_modules


def prepare_crop_frame(crop_vision_frame: VisionFrame) -> VisionFrame:
    crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
    crop_vision_frame = (crop_vision_frame - 0.5) / 0.5
    crop_vision_frame = numpy.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis=0).astype(numpy.float32)
    return crop_vision_frame


def normalize_crop_frame(crop_vision_frame: VisionFrame) -> VisionFrame:
    crop_vision_frame = numpy.clip(crop_vision_frame, -1, 1)
    crop_vision_frame = (crop_vision_frame + 1) / 2
    crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
    crop_vision_frame = (crop_vision_frame * 255.0).round()
    crop_vision_frame = crop_vision_frame.astype(numpy.uint8)[:, :, ::-1]
    return crop_vision_frame


def blend_frame(temp_vision_frame: VisionFrame, paste_vision_frame: VisionFrame) -> VisionFrame:
    face_enhancer_blend = 1 - (state_manager.get_item('face_enhancer_blend') / 100)
    temp_vision_frame = cv2.addWeighted(temp_vision_frame, face_enhancer_blend, paste_vision_frame,
                                        1 - face_enhancer_blend, 0)
    return temp_vision_frame


class FaceEnhancer(BaseProcessor):
    MODEL_SET: ModelSet = \
        {
            'codeformer':
                {
                    'hashes':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.hash',
                                    'path': resolve_relative_path('../.assets/models/codeformer.hash')
                                }
                        },
                    'sources':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx',
                                    'path': resolve_relative_path('../.assets/models/codeformer.onnx')
                                }
                        },
                    'template': 'ffhq_512',
                    'size': (512, 512)
                },
            'gfpgan_1.2':
                {
                    'hashes':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.2.hash',
                                    'path': resolve_relative_path('../.assets/models/gfpgan_1.2.hash')
                                }
                        },
                    'sources':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.2.onnx',
                                    'path': resolve_relative_path('../.assets/models/gfpgan_1.2.onnx')
                                }
                        },
                    'template': 'ffhq_512',
                    'size': (512, 512)
                },
            'gfpgan_1.3':
                {
                    'hashes':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.3.hash',
                                    'path': resolve_relative_path('../.assets/models/gfpgan_1.3.hash')
                                }
                        },
                    'sources':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.3.onnx',
                                    'path': resolve_relative_path('../.assets/models/gfpgan_1.3.onnx')
                                }
                        },
                    'template': 'ffhq_512',
                    'size': (512, 512)
                },
            'gfpgan_1.4':
                {
                    'hashes':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.hash',
                                    'path': resolve_relative_path('../.assets/models/gfpgan_1.4.hash')
                                }
                        },
                    'sources':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx',
                                    'path': resolve_relative_path('../.assets/models/gfpgan_1.4.onnx')
                                }
                        },
                    'template': 'ffhq_512',
                    'size': (512, 512)
                },
            'gpen_bfr_256':
                {
                    'hashes':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_256.hash',
                                    'path': resolve_relative_path('../.assets/models/gpen_bfr_256.hash')
                                }
                        },
                    'sources':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/gpen_bfr_256.onnx')
                                }
                        },
                    'template': 'arcface_128_v2',
                    'size': (256, 256)
                },
            'gpen_bfr_512':
                {
                    'hashes':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_512.hash',
                                    'path': resolve_relative_path('../.assets/models/gpen_bfr_512.hash')
                                }
                        },
                    'sources':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_512.onnx',
                                    'path': resolve_relative_path('../.assets/models/gpen_bfr_512.onnx')
                                }
                        },
                    'template': 'ffhq_512',
                    'size': (512, 512)
                },
            'gpen_bfr_1024':
                {
                    'hashes':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_1024.hash',
                                    'path': resolve_relative_path('../.assets/models/gpen_bfr_1024.hash')
                                }
                        },
                    'sources':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_1024.onnx',
                                    'path': resolve_relative_path('../.assets/models/gpen_bfr_1024.onnx')
                                }
                        },
                    'template': 'ffhq_512',
                    'size': (1024, 1024)
                },
            'gpen_bfr_2048':
                {
                    'hashes':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_2048.hash',
                                    'path': resolve_relative_path('../.assets/models/gpen_bfr_2048.hash')
                                }
                        },
                    'sources':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_2048.onnx',
                                    'path': resolve_relative_path('../.assets/models/gpen_bfr_2048.onnx')
                                }
                        },
                    'template': 'ffhq_512',
                    'size': (2048, 2048)
                },
            'restoreformer_plus_plus':
                {
                    'hashes':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/restoreformer_plus_plus.hash',
                                    'path': resolve_relative_path('../.assets/models/restoreformer_plus_plus.hash')
                                }
                        },
                    'sources':
                        {
                            'face_enhancer':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/restoreformer_plus_plus.onnx',
                                    'path': resolve_relative_path('../.assets/models/restoreformer_plus_plus.onnx')
                                }
                        },
                    'template': 'ffhq_512',
                    'size': (512, 512)
                }
        }

    model_key = "face_enhancer_model"
    priority = 12

    def register_args(self, program: ArgumentParser) -> None:
        group_processors = find_argument_group(program, "processors")
        if group_processors:
            group_processors.add_argument("--face-enhancer-model", help=wording.get("help.face_enhancer_model"),
                                          default=config.get_str_value("processors.face_enhancer_model", "gfpgan_1.4"),
                                          choices=self.list_models())
            group_processors.add_argument("--face-enhancer-blend", help=wording.get("help.face_enhancer_blend"),
                                          type=int,
                                          default=config.get_int_value("processors.face_enhancer_blend", "85"),
                                          choices=processors_choices.face_enhancer_blend_range,
                                          metavar=create_int_metavar(processors_choices.face_enhancer_blend_range))
            job_store.register_step_keys(["face_enhancer_model", "face_enhancer_blend"])

    def apply_args(self, args: Args, apply_state_item: ApplyStateItem) -> None:
        apply_state_item("face_enhancer_model", args.get("face_enhancer_model"))
        apply_state_item("face_enhancer_blend", args.get("face_enhancer_blend"))

    def pre_process(self, mode: ProcessMode) -> bool:
        if mode in ["output", "preview"] and not (is_image(state_manager.get_item("target_path"))
                                                  or is_video(state_manager.get_item("target_path"))):
            logger.error(wording.get("choose_image_or_video_target"), __name__)
            return False
        if mode == "output" and not in_directory(state_manager.get_item("output_path")):
            logger.error(wording.get("specify_image_or_video_output"), __name__)
            return False
        if mode == "output" and not same_file_extension(
                [state_manager.get_item("target_path"), state_manager.get_item("output_path")]):
            logger.error(wording.get("match_target_and_output_extension"), __name__)
            return False
        return True

    def process_frame(self, inputs: FaceEnhancerInputs) -> VisionFrame:
        reference_faces = inputs.get("reference_faces")
        target_vision_frame = inputs.get("target_vision_frame")
        many_faces = sort_and_filter_faces(get_many_faces([target_vision_frame]))

        if state_manager.get_item("face_selector_mode") == "many":
            for target_face in many_faces:
                target_vision_frame = self.enhance_face(target_face, target_vision_frame)
        elif state_manager.get_item("face_selector_mode") == "one":
            target_face = get_one_face(many_faces)
            if target_face:
                target_vision_frame = self.enhance_face(target_face, target_vision_frame)
        elif state_manager.get_item("face_selector_mode") == "reference":
            for src_face_idx, ref_faces in reference_faces.items():
                similar_faces = find_similar_faces(many_faces, ref_faces, state_manager.get_item("reference_face_distance"))
                if similar_faces:
                    for similar_face in similar_faces:
                        target_vision_frame = self.enhance_face(similar_face, target_vision_frame)
        return target_vision_frame

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        reference_faces = (get_reference_faces() if state_manager.get_item(
            "face_selector_mode") == "reference" else (None, None))
        output_frames = []
        for queue_payload in process_manager.manage(queue_payloads):
            target_vision_path = queue_payload["frame_path"]
            target_vision_frame = read_image(target_vision_path)
            result_frame = self.process_frame({
                "reference_faces": reference_faces,
                "target_vision_frame": target_vision_frame,
            })
            write_image(target_vision_path, result_frame)
            output_frames.append((queue_payload["frame_number"], target_vision_path))
        return output_frames

    def process_image(self, target_path: str, output_path: str, reference_faces=None) -> None:
        if reference_faces is None:
            reference_faces = (
                get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else (None, None))
        target_vision_frame = read_static_image(target_path)
        output_vision_frame = self.process_frame({
            "reference_faces": reference_faces,
            "target_vision_frame": target_vision_frame,
        })
        write_image(output_path, output_vision_frame)

    def enhance_face(self, target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
        masker = FaceMasker()
        model_template = self.get_model_options().get('template')
        model_size = self.get_model_options().get('size')
        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame,
                                                                        target_face.landmark_set.get('5/68'),
                                                                        model_template, model_size)
        crop_mask = masker.create_combined_mask(
            state_manager.get_item('face_mask_types'),
            crop_vision_frame.shape[:2][::-1], 
            state_manager.get_item('face_mask_blur'),
            state_manager.get_item('face_mask_padding'),
            state_manager.get_item('face_mask_regions'),
            crop_vision_frame,
            temp_vision_frame,
            target_face.landmark_set.get('5/68'),
            target_face
        )

        crop_vision_frame = prepare_crop_frame(crop_vision_frame)
        crop_vision_frame = self.forward(crop_vision_frame)
        crop_vision_frame = normalize_crop_frame(crop_vision_frame)
        paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
        temp_vision_frame = blend_frame(temp_vision_frame, paste_vision_frame)
        return temp_vision_frame

    def forward(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        face_enhancer = self.get_inference_pool().get('face_enhancer')
        face_enhancer_inputs = {}

        for face_enhancer_input in face_enhancer.get_inputs():
            if face_enhancer_input.name == 'input':
                face_enhancer_inputs[face_enhancer_input.name] = crop_vision_frame
            if face_enhancer_input.name == 'weight':
                weight = numpy.array([1]).astype(numpy.double)
                face_enhancer_inputs[face_enhancer_input.name] = weight

        with thread_semaphore():
            crop_vision_frame = face_enhancer.run(None, face_enhancer_inputs)[0][0]

        return crop_vision_frame
    
    def forward_batch(self, crop_vision_frames: List[VisionFrame]) -> List[VisionFrame]:
        """Batch inference version of forward for improved throughput."""
        face_enhancer = self.get_inference_pool().get('face_enhancer')
        results = []
        
        for crop_vision_frame in crop_vision_frames:
            face_enhancer_inputs = {}
            for face_enhancer_input in face_enhancer.get_inputs():
                if face_enhancer_input.name == 'input':
                    face_enhancer_inputs[face_enhancer_input.name] = crop_vision_frame
                if face_enhancer_input.name == 'weight':
                    weight = numpy.array([1]).astype(numpy.double)
                    face_enhancer_inputs[face_enhancer_input.name] = weight

            with thread_semaphore():
                result = face_enhancer.run(None, face_enhancer_inputs)[0][0]
                results.append(result)
        
        return results

