from argparse import ArgumentParser
from typing import List, Tuple

import cv2
import numpy

from facefusion import config, logger, process_manager, state_manager, wording
from facefusion.audio import create_empty_audio_frame, get_voice_frame
from facefusion.common_helper import get_first
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.face_helper import create_bounding_box, paste_back, warp_face_by_bounding_box, \
    warp_face_by_face_landmark_5
from facefusion.face_selector import find_similar_faces, sort_and_filter_faces
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import filter_audio_paths, has_audio, in_directory, is_image, is_video, \
    resolve_relative_path, same_file_extension
from facefusion.jobs import job_store
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.typing import LipSyncerInputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.typing import ApplyStateItem, Args, AudioFrame, Face, ModelSet, \
    ProcessMode, QueuePayload, VisionFrame
from facefusion.vision import read_image, read_static_image, restrict_video_fps, write_image
from facefusion.workers.classes.face_masker import FaceMasker


def prepare_audio_frame(temp_audio_frame: AudioFrame) -> AudioFrame:
    temp_audio_frame = numpy.maximum(numpy.exp(-5 * numpy.log(10)), temp_audio_frame)
    temp_audio_frame = numpy.log10(temp_audio_frame) * 1.6 + 3.2
    temp_audio_frame = temp_audio_frame.clip(-4, 4).astype(numpy.float32)
    return numpy.expand_dims(temp_audio_frame, axis=(0, 1))


def prepare_crop_frame(crop_vision_frame: VisionFrame) -> VisionFrame:
    crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0)
    prepare_vision_frame = crop_vision_frame.copy()
    prepare_vision_frame[:, 48:] = 0
    crop_vision_frame = numpy.concatenate((prepare_vision_frame, crop_vision_frame), axis=3)
    return crop_vision_frame.transpose(0, 3, 1, 2).astype('float32') / 255.0


def normalize_close_frame(crop_vision_frame: VisionFrame) -> VisionFrame:
    crop_vision_frame = crop_vision_frame[0].transpose(1, 2, 0)
    crop_vision_frame = crop_vision_frame.clip(0, 1) * 255
    return crop_vision_frame.astype(numpy.uint8)


class LipSyncer(BaseProcessor):
    MODEL_SET: ModelSet = {
        'wav2lip_96': {
            'hashes': {
                'lip_syncer': {
                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/wav2lip_96.hash',
                    'path': resolve_relative_path('../.assets/models/wav2lip_96.hash')
                }
            },
            'sources': {
                'lip_syncer': {
                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/wav2lip_96.onnx',
                    'path': resolve_relative_path('../.assets/models/wav2lip_96.onnx')
                }
            },
            'size': (96, 96)
        },
        'wav2lip_gan_96': {
            'hashes': {
                'lip_syncer': {
                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/wav2lip_gan_96.hash',
                    'path': resolve_relative_path('../.assets/models/wav2lip_gan_96.hash')
                }
            },
            'sources': {
                'lip_syncer': {
                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/wav2lip_gan_96.onnx',
                    'path': resolve_relative_path('../.assets/models/wav2lip_gan_96.onnx')
                }
            },
            'size': (96, 96)
        }
    }

    model_key: str = 'lip_syncer_model'
    priority = 10

    def register_args(self, program: ArgumentParser) -> None:
        group_processors = find_argument_group(program, 'processors')
        if group_processors:
            group_processors.add_argument('--lip-syncer-model', help=wording.get('help.lip_syncer_model'),
                                          default=config.get_str_value('processors.lip_syncer_model', 'wav2lip_gan_96'),
                                          choices=self.list_models())
            job_store.register_step_keys(['lip_syncer_model'])

    def apply_args(self, args: Args, apply_state_item: ApplyStateItem) -> None:
        apply_state_item('lip_syncer_model', args.get('lip_syncer_model'))

    def pre_process(self, mode: ProcessMode) -> bool:
        if not has_audio(state_manager.get_item('source_paths')):
            logger.error(wording.get('choose_audio_source') + wording.get('exclamation_mark'), __name__)
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
        return True

    def sync_lip(self, target_face: Face, temp_audio_frame: AudioFrame, temp_vision_frame: VisionFrame) -> VisionFrame:
        masker = FaceMasker()
        model_size = self.get_model_options().get('size')
        temp_audio_frame = prepare_audio_frame(temp_audio_frame)
        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
            temp_vision_frame, target_face.landmark_set.get('5/68'), 'ffhq_512', (512, 512))
        face_landmark_68 = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(
            -1, 2)
        bounding_box = create_bounding_box(face_landmark_68)
        bounding_box[1] -= numpy.abs(bounding_box[3] - bounding_box[1]) * 0.125
        mouth_mask = masker.create_mouth_mask(face_landmark_68)
        box_mask = masker.create_static_box_mask(
            crop_vision_frame.shape[:2][::-1], state_manager.get_item('face_mask_blur'),
            state_manager.get_item('face_mask_padding'))
        crop_masks = [mouth_mask, box_mask]

        if 'occlusion' in state_manager.get_item('face_mask_types'):
            occlusion_mask = masker.create_occlusion_mask(crop_vision_frame)
            crop_masks.append(occlusion_mask)

        close_vision_frame, close_matrix = warp_face_by_bounding_box(crop_vision_frame, bounding_box, model_size)
        close_vision_frame = prepare_crop_frame(close_vision_frame)
        close_vision_frame = self.forward(temp_audio_frame, close_vision_frame)
        close_vision_frame = normalize_close_frame(close_vision_frame)
        crop_vision_frame = cv2.warpAffine(close_vision_frame, cv2.invertAffineTransform(close_matrix), (512, 512),
                                           borderMode=cv2.BORDER_REPLICATE)
        crop_mask = numpy.minimum.reduce(crop_masks)
        return paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)

    def forward(self, temp_audio_frame: AudioFrame, close_vision_frame: VisionFrame) -> VisionFrame:
        lip_syncer = self.get_inference_pool().get('lip_syncer')
        with conditional_thread_semaphore():
            return lip_syncer.run(None, {'source': temp_audio_frame, 'target': close_vision_frame})[0]

    def process_frame(self, inputs: LipSyncerInputs) -> VisionFrame:
        reference_faces = inputs.get('reference_faces')
        reference_faces_2 = inputs.get('reference_faces_2')
        source_audio_frame = inputs.get('source_audio_frame')
        source_audio_frame_2 = inputs.get('source_audio_frame_2')
        target_vision_frame = inputs.get('target_vision_frame')
        many_faces = sort_and_filter_faces(get_many_faces([target_vision_frame]))

        if state_manager.get_item('face_selector_mode') == 'many':
            if many_faces:
                for target_face in many_faces:
                    target_vision_frame = self.sync_lip(target_face, source_audio_frame, target_vision_frame)
        if state_manager.get_item('face_selector_mode') == 'one':
            target_face = get_one_face(many_faces)
            if target_face:
                target_vision_frame = self.sync_lip(target_face, source_audio_frame, target_vision_frame)
        if state_manager.get_item('face_selector_mode') == 'reference':
            for ref_face, src_audio in [(reference_faces, source_audio_frame),
                                        (reference_faces_2, source_audio_frame_2)]:
                similar_faces = find_similar_faces(many_faces, ref_face,
                                                   state_manager.get_item('reference_face_distance'))
                if similar_faces:
                    for similar_face in similar_faces:
                        target_vision_frame = self.sync_lip(similar_face, src_audio, target_vision_frame)
        return target_vision_frame

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        source_paths = state_manager.get_item('source_paths')
        source_paths_2 = state_manager.get_item('source_paths_2')
        source_audio_path = get_first(filter_audio_paths(source_paths))
        source_audio_path_2 = get_first(filter_audio_paths(source_paths_2))
        temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'),
                                            state_manager.get_item('output_video_fps'))
        output_frames = []

        for queue_payload in process_manager.manage(queue_payloads):
            target_vision_path = queue_payload['frame_path']
            frame_number = queue_payload['frame_number']
            source_audio_frame = get_voice_frame(source_audio_path, temp_video_fps, frame_number)
            source_audio_frame_2 = get_voice_frame(source_audio_path_2, temp_video_fps, frame_number)
            reference_faces = queue_payload['reference_faces']
            reference_faces_2 = queue_payload['reference_faces_2']

            if not numpy.any(source_audio_frame):
                source_audio_frame = create_empty_audio_frame()
            if not numpy.any(source_audio_frame_2):
                source_audio_frame_2 = create_empty_audio_frame()
            target_vision_frame = read_image(target_vision_path)
            result_frame = self.process_frame(
                {
                    'reference_faces': reference_faces,
                    'reference_faces_2': reference_faces_2,
                    'source_audio_frame': source_audio_frame,
                    'source_audio_frame_2': source_audio_frame_2,
                    'target_vision_frame': target_vision_frame
                })
            write_image(target_vision_path, result_frame)
            output_frames.append((frame_number, target_vision_path))
        return output_frames

    def process_image(self, target_path: str, output_path: str) -> None:
        reference_faces, reference_faces_2 = (
            get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else (None, None)
        )
        source_paths = state_manager.get_item('source_paths')
        source_audio_path = get_first(filter_audio_paths(source_paths))
        source_audio_frame = get_voice_frame(source_audio_path, 25)
        target_vision_frame = read_static_image(target_path)
        result_frame = self.process_frame(
            {
                'reference_faces': reference_faces,
                'reference_faces_2': reference_faces_2,
                'source_audio_frame': source_audio_frame,
                'target_vision_frame': target_vision_frame
            })
        write_image(output_path, result_frame)
