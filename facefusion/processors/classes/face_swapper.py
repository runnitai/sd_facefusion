import os
from argparse import ArgumentParser
from typing import List, Tuple

import numpy

from facefusion import config, inference_manager, logger, process_manager, state_manager, wording
from facefusion.common_helper import get_first
from facefusion.execution import has_execution_provider
from facefusion.face_analyser import get_many_faces, get_one_face, get_average_faces
from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
from facefusion.face_selector import find_similar_faces, sort_and_filter_faces
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import has_image, in_directory, is_image, is_video, \
    resolve_relative_path, same_file_extension
from facefusion.inference_manager import get_static_model_initializer
from facefusion.jobs import job_store
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.pixel_boost import explode_pixel_boost, implode_pixel_boost
from facefusion.processors.typing import FaceSwapperInputs
from facefusion.program_helper import find_argument_group, suggest_face_swapper_pixel_boost_choices
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.typing import ApplyStateItem, Args, Embedding, Face, InferencePool, ModelOptions, ModelSet, ProcessMode, \
    QueuePayload, VisionFrame, Padding
from facefusion.vision import read_image, read_static_image, unpack_resolution, write_image
from facefusion.workers.classes.face_masker import FaceMasker
from facefusion.workers.core import clear_worker_modules


def update_padding(padding: Padding, frame_number: int) -> Padding:
    if frame_number == -1:
        return padding

    disabled_times = state_manager.get_item('mask_disabled_times')
    enabled_times = state_manager.get_item('mask_enabled_times')

    latest_disabled_frame = max([frame for frame in disabled_times if frame <= frame_number], default=None)
    latest_enabled_frame = max([frame for frame in enabled_times if frame <= frame_number], default=None)

    if latest_disabled_frame is not None and (
            latest_enabled_frame is None or latest_disabled_frame > latest_enabled_frame):
        new_padding = (0, 0, 0, 0)
        return new_padding
    return padding


class FaceSwapper(BaseProcessor):
    MODEL_SET: ModelSet = \
        {
            'blendswap_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/blendswap_256.hash',
                                    'path': resolve_relative_path('../.assets/models/blendswap_256.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/blendswap_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/blendswap_256.onnx')
                                }
                        },
                    'type': 'blendswap',
                    'template': 'ffhq_512',
                    'size': (256, 256),
                    'mean': [0.0, 0.0, 0.0],
                    'standard_deviation': [1.0, 1.0, 1.0]
                },
            'ghost_1_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_1_256.hash',
                                    'path': resolve_relative_path('../.assets/models/ghost_1_256.hash')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_converter_ghost.hash',
                                    'path': resolve_relative_path('../.assets/models/arcface_converter_ghost.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_1_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/ghost_1_256.onnx')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_converter_ghost.onnx',
                                    'path': resolve_relative_path('../.assets/models/arcface_converter_ghost.onnx')
                                }
                        },
                    'type': 'ghost',
                    'template': 'arcface_112_v1',
                    'size': (256, 256),
                    'mean': [0.5, 0.5, 0.5],
                    'standard_deviation': [0.5, 0.5, 0.5]
                },
            'ghost_2_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_2_256.hash',
                                    'path': resolve_relative_path('../.assets/models/ghost_2_256.hash')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_converter_ghost.hash',
                                    'path': resolve_relative_path('../.assets/models/arcface_converter_ghost.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_2_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/ghost_2_256.onnx')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_converter_ghost.onnx',
                                    'path': resolve_relative_path('../.assets/models/arcface_converter_ghost.onnx')
                                }
                        },
                    'type': 'ghost',
                    'template': 'arcface_112_v1',
                    'size': (256, 256),
                    'mean': [0.5, 0.5, 0.5],
                    'standard_deviation': [0.5, 0.5, 0.5]
                },
            'ghost_3_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_3_256.hash',
                                    'path': resolve_relative_path('../.assets/models/ghost_3_256.hash')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_converter_ghost.hash',
                                    'path': resolve_relative_path('../.assets/models/arcface_converter_ghost.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_3_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/ghost_3_256.onnx')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_converter_ghost.onnx',
                                    'path': resolve_relative_path('../.assets/models/arcface_converter_ghost.onnx')
                                }
                        },
                    'type': 'ghost',
                    'template': 'arcface_112_v1',
                    'size': (256, 256),
                    'mean': [0.5, 0.5, 0.5],
                    'standard_deviation': [0.5, 0.5, 0.5]
                },
            'inswapper_128':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.hash',
                                    'path': resolve_relative_path('../.assets/models/inswapper_128.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx',
                                    'path': resolve_relative_path('../.assets/models/inswapper_128.onnx')
                                }
                        },
                    'type': 'inswapper',
                    'template': 'arcface_128_v2',
                    'size': (128, 128),
                    'mean': [0.0, 0.0, 0.0],
                    'standard_deviation': [1.0, 1.0, 1.0]
                },
            'inswapper_128_fp16':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.hash',
                                    'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.onnx',
                                    'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.onnx')
                                }
                        },
                    'type': 'inswapper',
                    'template': 'arcface_128_v2',
                    'size': (128, 128),
                    'mean': [0.0, 0.0, 0.0],
                    'standard_deviation': [1.0, 1.0, 1.0]
                },
            'simswap_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_256.hash',
                                    'path': resolve_relative_path('../.assets/models/simswap_256.hash')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_converter_simswap.hash',
                                    'path': resolve_relative_path('../.assets/models/arcface_converter_simswap.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/simswap_256.onnx')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_converter_simswap.onnx',
                                    'path': resolve_relative_path('../.assets/models/arcface_converter_simswap.onnx')
                                }
                        },
                    'type': 'simswap',
                    'template': 'arcface_112_v1',
                    'size': (256, 256),
                    'mean': [0.485, 0.456, 0.406],
                    'standard_deviation': [0.229, 0.224, 0.225]
                },
            'simswap_unofficial_512':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_unofficial_512.hash',
                                    'path': resolve_relative_path('../.assets/models/simswap_unofficial_512.hash')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_converter_simswap.hash',
                                    'path': resolve_relative_path('../.assets/models/arcface_converter_simswap.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_unofficial_512.onnx',
                                    'path': resolve_relative_path('../.assets/models/simswap_unofficial_512.onnx')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_converter_simswap.onnx',
                                    'path': resolve_relative_path('../.assets/models/arcface_converter_simswap.onnx')
                                }
                        },
                    'type': 'simswap',
                    'template': 'arcface_112_v1',
                    'size': (512, 512),
                    'mean': [0.0, 0.0, 0.0],
                    'standard_deviation': [1.0, 1.0, 1.0]
                },
            'uniface_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/uniface_256.hash',
                                    'path': resolve_relative_path('../.assets/models/uniface_256.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/uniface_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/uniface_256.onnx')
                                }
                        },
                    'type': 'uniface',
                    'template': 'ffhq_512',
                    'size': (256, 256),
                    'mean': [0.5, 0.5, 0.5],
                    'standard_deviation': [0.5, 0.5, 0.5]
                }
        }

    model_key: str = 'face_swapper_model'
    priority: int = 0
    preload: bool = True
    preferred_provider = 'cuda'
    src_cache = {}

    def register_args(self, program: ArgumentParser) -> None:
        group_processors = find_argument_group(program, 'processors')
        if group_processors:
            group_processors.add_argument('--face-swapper-model', help=wording.get('help.face_swapper_model'),
                                          default=config.get_str_value('processors.face_swapper_model',
                                                                       'inswapper_128_fp16'),
                                          choices=self.list_models())
            face_swapper_pixel_boost_choices = suggest_face_swapper_pixel_boost_choices(program)
            group_processors.add_argument('--face-swapper-pixel-boost',
                                          help=wording.get('help.face_swapper_pixel_boost'),
                                          default=config.get_str_value('processors.face_swapper_pixel_boost',
                                                                       get_first(face_swapper_pixel_boost_choices)),
                                          choices=face_swapper_pixel_boost_choices)
            job_store.register_step_keys(['face_swapper_model', 'face_swapper_pixel_boost'])

    def apply_args(self, args: Args, apply_state_item: ApplyStateItem) -> None:
        apply_state_item('face_swapper_model', args.get('face_swapper_model'))
        apply_state_item('face_swapper_pixel_boost', args.get('face_swapper_pixel_boost'))

    def pre_process(self, mode: ProcessMode) -> bool:
        self.src_cache = {}
        source_paths = state_manager.get_item('source_paths')
        source_paths_2 = state_manager.get_item('source_paths_2')
        if not has_image(source_paths) and not has_image(source_paths_2):
            logger.error(wording.get('choose_image_source') + wording.get('exclamation_mark'), __name__)
            return False
        source_faces = get_average_faces()
        source_face_values = [value for value in source_faces.values()]
        target_folder = state_manager.get_item('target_folder')
        is_batch = False
        if target_folder is not None and target_folder != "" and os.path.isdir(target_folder):
            is_batch = True
            logger.info("Batch processing is enabled", __name__)
        if not len(source_face_values):
            logger.error(wording.get('no_source_face_detected') + wording.get('exclamation_mark'), __name__)
            return False
        if mode in ['output', 'preview'] and not is_image(state_manager.get_item('target_path')) and not is_video(
                state_manager.get_item('target_path')) and not is_batch:
            logger.error(wording.get('choose_image_or_video_target') + wording.get('exclamation_mark'), __name__)
            return False
        if mode == 'output' and not in_directory(state_manager.get_item('output_path')) and not is_batch:
            logger.error(wording.get('specify_image_or_video_output') + wording.get('exclamation_mark'), __name__)
            return False
        if mode == 'output' and not is_batch and not same_file_extension(
                [state_manager.get_item('target_path'), state_manager.get_item('output_path')]):
            logger.error(wording.get('match_target_and_output_extension') + wording.get('exclamation_mark'), __name__)
            return False
        return True

    def post_process(self) -> None:
        self.src_cache = {}
        if state_manager.get_item("video_memory_strategy") in ["strict", "moderate"]:
            self.clear_inference_pool()
        if state_manager.get_item("video_memory_strategy") == "strict":
            clear_worker_modules()

    def process_frame(self, inputs: FaceSwapperInputs) -> VisionFrame:
        reference_faces = inputs.get('reference_faces')
        source_faces = inputs.get('source_faces')
        face_selector_mode = state_manager.get_item('face_selector_mode')
        source_face = next(iter(source_faces.values())) if not face_selector_mode == 'reference' else None

        target_vision_frame = inputs.get('target_vision_frame')
        many_faces = sort_and_filter_faces(get_many_faces([target_vision_frame]))
        src_idx = 0
        if face_selector_mode == 'many':
            if many_faces:
                for target_face in many_faces:
                    target_vision_frame = self.swap_face(source_face, target_face, target_vision_frame, src_idx)
            else:
                print("No target face found")
        if face_selector_mode == 'one':
           # watch.next("one_face")
            target_face = get_one_face(many_faces)
            if target_face:
                target_vision_frame = self.swap_face(source_face, target_face, target_vision_frame, src_idx)
            else:
                logger.info("No target face found", __name__)
        if face_selector_mode == 'reference':
            # Make a unique set of keys from reference_faces and source_faces
            reference_face_keys = set(reference_faces.keys())
            source_face_keys = set(source_faces.keys())
            all_keys = reference_face_keys.union(source_face_keys)  # Combined set of keys

            for src_face_idx in all_keys:
                ref_faces = reference_faces.get(src_face_idx)
                src_face = source_faces.get(src_face_idx)

                if not ref_faces or not src_face:
                    continue

                similar_faces = find_similar_faces(
                    many_faces, ref_faces,
                    state_manager.get_item('reference_face_distance')
                )

                if similar_faces:
                    for similar_face in similar_faces:
                        target_vision_frame = self.swap_face(
                            src_face, similar_face, target_vision_frame, src_face_idx
                        )

        return target_vision_frame

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        output_frames = []
        for queue_payload in process_manager.manage(queue_payloads):
            target_vision_path = queue_payload['frame_path']
            target_frame_number = queue_payload['frame_number']
            source_faces = queue_payload['source_faces']
            reference_faces = queue_payload['reference_faces']
            target_vision_frame = read_image(target_vision_path)
            result_frame = self.process_frame(
                {
                    'reference_faces': reference_faces,
                    'source_faces': source_faces,
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
        source_faces = get_average_faces()
        target_vision_frame = read_static_image(target_path)
        result_frame = self.process_frame(
            {
                'reference_faces': reference_faces,
                'source_faces': source_faces,
                'target_vision_frame': target_vision_frame,
                'target_frame_number': -1
            })
        write_image(output_path, result_frame)

    def get_model_options(self) -> ModelOptions:
        face_swapper_model = state_manager.get_item(self.model_key)
        face_swapper_model = 'inswapper_128' if has_execution_provider(
            'coreml') and face_swapper_model == 'inswapper_128_fp16' else face_swapper_model
        return self.MODEL_SET.get(face_swapper_model)

    def get_inference_pool(self) -> InferencePool:
        model_sources = self.get_model_options().get('sources')
        model_context = __name__ + '.' + state_manager.get_item(self.model_key)
        return inference_manager.get_inference_pool(model_context, model_sources)

    def clear_inference_pool(self) -> None:
        model_context = __name__ + '.' + state_manager.get_item(self.model_key)
        inference_manager.clear_inference_pool(model_context)

    def swap_face(self, source_face: Face, target_face: Face, temp_vision_frame: VisionFrame,
                  src_idx: int) -> VisionFrame:
        masker = FaceMasker()
        model_template = self.get_model_options().get('template')
        model_size = self.get_model_options().get('size')
        pixel_boost_size = unpack_resolution(state_manager.get_item('face_swapper_pixel_boost'))
        pixel_boost_total = pixel_boost_size[0] // model_size[0]
        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame,
                                                                        target_face.landmark_set.get('5/68'),
                                                                        model_template, pixel_boost_size)
        temp_vision_frames = []
        
        # Use the combined mask function instead of creating individual masks
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

        pixel_boost_vision_frames = implode_pixel_boost(crop_vision_frame, pixel_boost_total, model_size)
        
        # Use multi-GPU batch processing for pixel boost frames when available
        face_swapper = self.get_inference_pool().get('face_swapper')
        from facefusion.inference_manager import MultiGPUInferenceWrapper
        
        if isinstance(face_swapper, MultiGPUInferenceWrapper) and len(pixel_boost_vision_frames) > 1:
            # Use true multi-GPU parallel processing
            logger.debug(f"Using multi-GPU parallel processing for {len(pixel_boost_vision_frames)} pixel boost frames", __name__)
            prepared_frames = [self.prepare_crop_frame(frame) for frame in pixel_boost_vision_frames]
            swapped_frames = self.forward_swap_face_batch(source_face, prepared_frames, src_idx)
            temp_vision_frames = [self.normalize_crop_frame(frame) for frame in swapped_frames]
            # Track metrics
            self.optimization_stats['multi_gpu_batches'] += 1
            self.optimization_stats['total_batch_size'] += len(pixel_boost_vision_frames)
        elif self.batch_scheduler and len(pixel_boost_vision_frames) > 1:
            # Use adaptive batching for single GPU
            def batch_swap_inference(frames_batch):
                results = []
                for frame in frames_batch:
                    prepared_frame = self.prepare_crop_frame(frame)
                    swapped_frame = self.forward_swap_face(source_face, prepared_frame, src_idx)
                    normalized_frame = self.normalize_crop_frame(swapped_frame)
                    results.append(normalized_frame)
                return results
            
            temp_vision_frames = self.process_with_adaptive_batching(
                pixel_boost_vision_frames, batch_swap_inference, mode="default"
            )
            # Track metrics
            self.optimization_stats['adaptive_batches'] += 1
            self.optimization_stats['total_batch_size'] += len(pixel_boost_vision_frames)
        else:
            # Process individually for small numbers of frames
            for pixel_boost_vision_frame in pixel_boost_vision_frames:
                pixel_boost_vision_frame = self.prepare_crop_frame(pixel_boost_vision_frame)
                pixel_boost_vision_frame = self.forward_swap_face(source_face, pixel_boost_vision_frame, src_idx)
                pixel_boost_vision_frame = self.normalize_crop_frame(pixel_boost_vision_frame)
                temp_vision_frames.append(pixel_boost_vision_frame)
                # Track metrics
                self.optimization_stats['single_operations'] += 1
                self.optimization_stats['total_batch_size'] += 1
        
        crop_vision_frame = explode_pixel_boost(temp_vision_frames, pixel_boost_total, model_size, pixel_boost_size)

        # No need to reduce crop_masks as we're using the combined mask
        crop_mask = crop_mask.clip(0, 1)
        temp_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
        return temp_vision_frame

    def forward_swap_face(self, source_face: Face, crop_vision_frame: VisionFrame, src_idx: int) -> VisionFrame:
        face_swapper = self.get_inference_pool().get('face_swapper')
        
        # Check if we have multi-GPU wrapper for parallel processing
        from facefusion.inference_manager import MultiGPUInferenceWrapper
        if isinstance(face_swapper, MultiGPUInferenceWrapper):
            # Use regular run method for individual frames - still benefits from round-robin load distribution
            model_type = self.get_model_options().get('type')
            face_swapper_inputs = {}

            for face_swapper_input in face_swapper.get_inputs():
                if face_swapper_input.name == 'source':
                    if model_type == 'blendswap' or model_type == 'uniface':
                        face_swapper_inputs[face_swapper_input.name] = self.prepare_source_frame(source_face, src_idx)
                    else:
                        face_swapper_inputs[face_swapper_input.name] = self.prepare_source_embedding(source_face, src_idx)
                if face_swapper_input.name == 'target':
                    face_swapper_inputs[face_swapper_input.name] = crop_vision_frame

            crop_vision_frame = face_swapper.run(None, face_swapper_inputs)[0][0]
        else:
            # Original single GPU path
            model_type = self.get_model_options().get('type')
            face_swapper_inputs = {}

            for face_swapper_input in face_swapper.get_inputs():
                if face_swapper_input.name == 'source':
                    if model_type == 'blendswap' or model_type == 'uniface':
                        face_swapper_inputs[face_swapper_input.name] = self.prepare_source_frame(source_face, src_idx)
                    else:
                        face_swapper_inputs[face_swapper_input.name] = self.prepare_source_embedding(source_face, src_idx)
                if face_swapper_input.name == 'target':
                    face_swapper_inputs[face_swapper_input.name] = crop_vision_frame

            with conditional_thread_semaphore():
                crop_vision_frame = face_swapper.run(None, face_swapper_inputs)[0][0]
        return crop_vision_frame

    def forward_swap_face_batch(self, source_face: Face, crop_vision_frames: List[VisionFrame], src_idx: int) -> List[VisionFrame]:
        """Process multiple frames in parallel across GPUs for true speedup."""
        face_swapper = self.get_inference_pool().get('face_swapper')
        
        # Check if we have multi-GPU wrapper for parallel processing
        from facefusion.inference_manager import MultiGPUInferenceWrapper
        if isinstance(face_swapper, MultiGPUInferenceWrapper) and len(crop_vision_frames) > 1:
            # Prepare all inputs for batch processing
            model_type = self.get_model_options().get('type')
            batch_inputs = []
            
            for crop_vision_frame in crop_vision_frames:
                face_swapper_inputs = {}
                for face_swapper_input in face_swapper.get_inputs():
                    if face_swapper_input.name == 'source':
                        if model_type == 'blendswap' or model_type == 'uniface':
                            face_swapper_inputs[face_swapper_input.name] = self.prepare_source_frame(source_face, src_idx)
                        else:
                            face_swapper_inputs[face_swapper_input.name] = self.prepare_source_embedding(source_face, src_idx)
                    if face_swapper_input.name == 'target':
                        face_swapper_inputs[face_swapper_input.name] = crop_vision_frame
                batch_inputs.append(face_swapper_inputs)
            
            # Process batch in parallel across GPUs
            batch_results = face_swapper.run_batch_parallel(None, batch_inputs)
            return [result[0][0] for result in batch_results]
        else:
            # Fallback to single processing
            results = []
            for crop_vision_frame in crop_vision_frames:
                result = self.forward_swap_face(source_face, crop_vision_frame, src_idx)
                results.append(result)
            return results

    def prepare_source_frame(self, source_face: Face, src_idx: int) -> VisionFrame:
        if src_idx in self.src_cache:
            return self.src_cache[src_idx]
        model_type = self.get_model_options().get('type')
        source_vision_frame = read_static_image(get_first(state_manager.get_item('source_paths')))

        if model_type == 'blendswap':
            source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame,
                                                                  source_face.landmark_set.get('5/68'),
                                                                  'arcface_112_v2', (112, 112))
        if model_type == 'uniface':
            source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame,
                                                                  source_face.landmark_set.get('5/68'),
                                                                  'ffhq_512', (256, 256))
        source_vision_frame = source_vision_frame[:, :, ::-1] / 255.0
        source_vision_frame = source_vision_frame.transpose(2, 0, 1)
        source_vision_frame = numpy.expand_dims(source_vision_frame, axis=0).astype(numpy.float32)
        self.src_cache[src_idx] = source_vision_frame
        return source_vision_frame

    def prepare_source_embedding(self, source_face: Face, src_idx) -> Embedding:
        if src_idx in self.src_cache:
            return self.src_cache[src_idx]
        model_type = self.get_model_options().get('type')

        if model_type == 'ghost':
            source_embedding, _ = self.convert_embedding(source_face)
            source_embedding = source_embedding.reshape(1, -1)
        elif model_type == 'inswapper':
            model_path = self.get_model_options().get('sources').get('face_swapper').get('path')
            model_initializer = get_static_model_initializer(model_path)
            source_embedding = source_face.embedding.reshape((1, -1))
            source_embedding = numpy.dot(source_embedding, model_initializer) / numpy.linalg.norm(source_embedding)
        else:
            _, source_normed_embedding = self.convert_embedding(source_face)
            source_embedding = source_normed_embedding.reshape(1, -1)
        self.src_cache[src_idx] = source_embedding
        return source_embedding

    def convert_embedding(self, source_face: Face) -> Tuple[Embedding, Embedding]:
        embedding = source_face.embedding.reshape(-1, 512)
        embedding = self.forward_convert_embedding(embedding)
        embedding = embedding.ravel()
        normed_embedding = embedding / numpy.linalg.norm(embedding)
        return embedding, normed_embedding

    def forward_convert_embedding(self, embedding: Embedding) -> Embedding:
        embedding_converter = self.get_inference_pool().get('embedding_converter')

        with conditional_thread_semaphore():
            embedding = embedding_converter.run(None,
                                                {
                                                    'input': embedding
                                                })[0]

        return embedding

    def prepare_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        model_mean = self.get_model_options().get('mean')
        model_standard_deviation = self.get_model_options().get('standard_deviation')

        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        crop_vision_frame = (crop_vision_frame - model_mean) / model_standard_deviation
        crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
        crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0).astype(numpy.float32)
        return crop_vision_frame

    def normalize_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        model_type = self.get_model_options().get('type')
        model_mean = self.get_model_options().get('mean')
        model_standard_deviation = self.get_model_options().get('standard_deviation')

        crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
        if model_type == 'ghost' or model_type == 'uniface':
            crop_vision_frame = crop_vision_frame * model_standard_deviation + model_mean
        crop_vision_frame = crop_vision_frame.clip(0, 1)
        crop_vision_frame = crop_vision_frame[:, :, ::-1] * 255
        return crop_vision_frame
