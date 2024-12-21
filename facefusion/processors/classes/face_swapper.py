from argparse import ArgumentParser
from typing import List, Tuple

import numpy

from facefusion import config, inference_manager, logger, process_manager, state_manager, wording
from facefusion.common_helper import get_first
from facefusion.execution import has_execution_provider
from facefusion.face_analyser import get_many_faces, get_one_face, get_avg_faces
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
    preferred_provider = 'tensorrt'
    model_options: ModelOptions = {}
    model_type: str = ''
    model_template: str = ''
    model_size: Tuple[int, int] = (0, 0)
    model_mean: List[float] = []
    model_std: List[float] = []
    face_swapper: InferencePool = {}
    embedding_converter: InferencePool = {}
    inference_pool: InferencePool = {}
    face_selector_mode: str = ''
    reference_face_distance: float = 0.0
    face_mask_types: List[str] = []
    face_mask_blur: float = 0.0
    face_mask_regions: List[str] = []
    face_mask_padding: Padding = (0, 0, 0, 0)
    pixel_boost_value: str = ''
    source_face: Face = None
    source_face_2: Face = None
    prepared_source_input: VisionFrame = None
    prepared_source_input_2: VisionFrame = None

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
        source_paths = state_manager.get_item('source_paths')
        source_paths_2 = state_manager.get_item('source_paths_2')
        if not has_image(source_paths) and not has_image(source_paths_2):
            logger.error(wording.get('choose_image_source') + wording.get('exclamation_mark'), __name__)
            return False
        source_faces, source_faces_2 = get_avg_faces()
        if not get_one_face([source_faces]) and not get_one_face([source_faces_2]):
            logger.error(wording.get('no_source_face_detected') + wording.get('exclamation_mark'), __name__)
            return False
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

        # Cache model options and inference pool
        self.model_options = self.get_model_options()
        self.model_type = self.model_options.get('type')
        self.model_template = self.model_options.get('template')
        self.model_size = self.model_options.get('size')
        self.model_mean = self.model_options.get('mean')
        self.model_std = self.model_options.get('standard_deviation')

        self.inference_pool = self.get_inference_pool()
        self.face_swapper = self.inference_pool.get('face_swapper')
        self.embedding_converter = self.inference_pool.get(
            'embedding_converter') if 'embedding_converter' in self.model_options.get('sources', {}) else None

        self.face_selector_mode = state_manager.get_item('face_selector_mode')
        self.reference_face_distance = state_manager.get_item('reference_face_distance')
        self.face_mask_types = state_manager.get_item('face_mask_types')
        self.face_mask_blur = state_manager.get_item('face_mask_blur')
        self.face_mask_regions = state_manager.get_item('face_mask_regions')
        self.face_mask_padding = state_manager.get_item('face_mask_padding')
        self.pixel_boost_value = state_manager.get_item('face_swapper_pixel_boost')

        # Store source faces
        self.source_face = get_one_face([source_faces])
        self.source_face_2 = get_one_face([source_faces_2])

        # Precompute source inputs to avoid repetitive disk reads
        self.prepared_source_input = None
        self.prepared_source_input_2 = None

        def prepare_source_input(source_face, source_paths):
            if source_face is None or not has_image(source_paths):
                return None
            # Read the source image once
            source_vision_frame = read_static_image(get_first(source_paths))

            # Prepare input depending on model_type
            if self.model_type in ['blendswap', 'uniface']:
                # frame-based preparation
                return self._prepare_source_frame_once(source_face, source_vision_frame)
            else:
                # embedding-based preparation
                return self._prepare_source_embedding_once(source_face)

        self.prepared_source_input = prepare_source_input(self.source_face, source_paths)
        self.prepared_source_input_2 = prepare_source_input(self.source_face_2, source_paths_2)

        return True

    def _prepare_source_frame_once(self, source_face: Face, source_vision_frame: VisionFrame) -> VisionFrame:
        # This mirrors logic from prepare_source_frame but without reading from disk again
        if self.model_type == 'blendswap':
            source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame,
                                                                  source_face.landmark_set.get('5/68'),
                                                                  'arcface_112_v2', (112, 112))
        if self.model_type == 'uniface':
            source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame,
                                                                  source_face.landmark_set.get('5/68'),
                                                                  'ffhq_512', (256, 256))
        source_vision_frame = source_vision_frame[:, :, ::-1] / 255.0
        source_vision_frame = source_vision_frame.transpose(2, 0, 1)
        source_vision_frame = numpy.expand_dims(source_vision_frame, axis=0).astype(numpy.float32)
        return source_vision_frame

    def _prepare_source_embedding_once(self, source_face: Face) -> Embedding:
        # Same logic from prepare_source_embedding
        if self.model_type == 'ghost':
            source_embedding, _ = self.convert_embedding(source_face)
            source_embedding = source_embedding.reshape(1, -1)
        elif self.model_type == 'inswapper':
            model_path = self.model_options.get('sources').get('face_swapper').get('path')
            model_initializer = get_static_model_initializer(model_path)
            source_embedding = source_face.embedding.reshape((1, -1))
            source_embedding = numpy.dot(source_embedding, model_initializer) / numpy.linalg.norm(source_embedding)
        else:
            _, source_normed_embedding = self.convert_embedding(source_face)
            source_embedding = source_normed_embedding.reshape(1, -1)
        return source_embedding

    def process_frame(self, inputs: FaceSwapperInputs) -> VisionFrame:
        reference_faces = inputs.get('reference_faces')
        reference_faces_2 = inputs.get('reference_faces_2')
        source_face = inputs.get('source_face')
        source_face_2 = inputs.get('source_face_2')
        target_vision_frame = inputs.get('target_vision_frame')
        many_faces = sort_and_filter_faces(get_many_faces([target_vision_frame]))

        if self.face_selector_mode == 'many':
            if many_faces:
                for target_face in many_faces:
                    target_vision_frame = self.swap_face(source_face, target_face, target_vision_frame)

        if self.face_selector_mode == 'one':
            target_face = get_one_face(many_faces)
            if target_face:
                target_vision_frame = self.swap_face(source_face, target_face, target_vision_frame)

        if self.face_selector_mode == 'reference':
            for ref_faces, src_face in [(reference_faces, source_face), (reference_faces_2, source_face_2)]:
                if not ref_faces or not src_face:
                    continue
                similar_faces = find_similar_faces(many_faces, ref_faces, self.reference_face_distance)
                if similar_faces:
                    for similar_face in similar_faces:
                        target_vision_frame = self.swap_face(src_face, similar_face, target_vision_frame)

        return target_vision_frame

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        output_frames = []
        for queue_payload in process_manager.manage(queue_payloads):
            target_vision_path = queue_payload['frame_path']
            target_frame_number = queue_payload['frame_number']
            source_face = queue_payload['source_face']
            source_face_2 = queue_payload['source_face_2']
            reference_faces = queue_payload['reference_faces']
            reference_faces_2 = queue_payload['reference_faces_2']
            target_vision_frame = read_image(target_vision_path)
            result_frame = self.process_frame(
                {
                    'reference_faces': reference_faces,
                    'reference_faces_2': reference_faces_2,
                    'source_face': source_face,
                    'source_face_2': source_face_2,
                    'target_vision_frame': target_vision_frame,
                    'target_frame_number': target_frame_number
                })
            write_image(target_vision_path, result_frame)
            output_frames.append((target_frame_number, target_vision_path))
        return output_frames

    def process_image(self, target_path: str, output_path: str) -> None:
        reference_faces, reference_faces_2 = (
            get_reference_faces(True) if 'reference' in self.face_selector_mode else (None, None))
        source_face, source_face_2 = get_avg_faces()
        target_vision_frame = read_static_image(target_path)
        result_frame = self.process_frame(
            {
                'reference_faces': reference_faces,
                'reference_faces_2': reference_faces_2,
                'source_face': source_face,
                'source_face_2': source_face_2,
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
        model_sources = self.model_options.get('sources')
        model_context = __name__ + '.' + state_manager.get_item(self.model_key)
        return inference_manager.get_inference_pool(model_context, model_sources)

    def clear_inference_pool(self) -> None:
        model_context = __name__ + '.' + state_manager.get_item(self.model_key)
        inference_manager.clear_inference_pool(model_context)

    def swap_face(self, source_face: Face, target_face: Face, temp_vision_frame: VisionFrame,
                  frame_number=-1) -> VisionFrame:

        masker = FaceMasker()
        model_template = self.model_template
        model_size = self.model_size
        pixel_boost_size = unpack_resolution(self.pixel_boost_value)
        pixel_boost_total = pixel_boost_size[0] // model_size[0]

        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
            temp_vision_frame,
            target_face.landmark_set.get('5/68'),
            model_template, pixel_boost_size
        )

        temp_vision_frames = []
        crop_masks = []
        padding = update_padding(self.face_mask_padding, frame_number)

        if 'box' in self.face_mask_types:
            box_mask = masker.create_static_box_mask(
                crop_vision_frame.shape[:2][::-1],
                self.face_mask_blur,
                padding
            )
            crop_masks.append(box_mask)

        if 'occlusion' in self.face_mask_types:
            occlusion_mask = masker.create_occlusion_mask(crop_vision_frame)
            crop_masks.append(occlusion_mask)

        pixel_boost_vision_frames = implode_pixel_boost(
            crop_vision_frame, pixel_boost_total, model_size
        )
        if len(pixel_boost_vision_frames) > 1:
            for pixel_boost_vision_frame in pixel_boost_vision_frames:
                pixel_boost_vision_frame = self.prepare_crop_frame(pixel_boost_vision_frame)
                pixel_boost_vision_frame = self.forward_swap_face(source_face, pixel_boost_vision_frame)
                pixel_boost_vision_frame = self.normalize_crop_frame(pixel_boost_vision_frame)
                temp_vision_frames.append(pixel_boost_vision_frame)

            crop_vision_frame = explode_pixel_boost(
                temp_vision_frames, pixel_boost_total, model_size, pixel_boost_size
            )
        else:
            crop_vision_frame = self.prepare_crop_frame(crop_vision_frame)
            crop_vision_frame = self.forward_swap_face(source_face, crop_vision_frame)
            crop_vision_frame = self.normalize_crop_frame(crop_vision_frame)

        if 'region' in self.face_mask_types:
            region_mask = masker.create_region_mask(
                crop_vision_frame, self.face_mask_regions
            )
            crop_masks.append(region_mask)

        crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
        temp_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)

        return temp_vision_frame

    def forward_swap_face(self, source_face: Face, crop_vision_frame: VisionFrame) -> VisionFrame:
        # Instead of preparing each time, we use precomputed inputs
        if source_face is self.source_face:
            source_input = self.prepared_source_input
        elif source_face is self.source_face_2:
            source_input = self.prepared_source_input_2
        else:
            # fallback (should not happen if code is correct)
            source_input = None

        face_swapper_inputs = {}

        for face_swapper_input in self.face_swapper.get_inputs():
            if face_swapper_input.name == 'source':
                face_swapper_inputs[face_swapper_input.name] = source_input
            if face_swapper_input.name == 'target':
                face_swapper_inputs[face_swapper_input.name] = crop_vision_frame

        with conditional_thread_semaphore():
            crop_vision_frame = self.face_swapper.run(None, face_swapper_inputs)[0][0]
        return crop_vision_frame

    def prepare_source_frame(self, source_face: Face, source_vision_frame: VisionFrame) -> VisionFrame:
        # Not used directly anymore, but kept for reference
        # Moved logic to _prepare_source_frame_once
        pass

    def prepare_source_embedding(self, source_face: Face) -> Embedding:
        # Not used directly anymore, logic moved to _prepare_source_embedding_once
        pass

    def convert_embedding(self, source_face: Face) -> Tuple[Embedding, Embedding]:
        embedding = source_face.embedding.reshape(-1, 512)
        embedding = self.forward_convert_embedding(embedding)
        embedding = embedding.ravel()
        normed_embedding = embedding / numpy.linalg.norm(embedding)
        return embedding, normed_embedding

    def forward_convert_embedding(self, embedding: Embedding) -> Embedding:
        if not self.embedding_converter:
            return embedding
        with conditional_thread_semaphore():
            embedding = self.embedding_converter.run(None,
                                                     {
                                                         'input': embedding
                                                     })[0]

        return embedding

    def prepare_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        crop_vision_frame = (crop_vision_frame - self.model_mean) / self.model_std
        crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
        crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0).astype(numpy.float32)
        return crop_vision_frame

    def normalize_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
        if self.model_type == 'ghost' or self.model_type == 'uniface':
            crop_vision_frame = crop_vision_frame * self.model_std + self.model_mean
        crop_vision_frame = crop_vision_frame.clip(0, 1)
        crop_vision_frame = crop_vision_frame[:, :, ::-1] * 255
        return crop_vision_frame
