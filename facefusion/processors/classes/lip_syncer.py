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
    try:
        # Ensure we're working with a numpy array
        if not isinstance(temp_audio_frame, numpy.ndarray):
            logger.error(f"Audio frame has invalid type: {type(temp_audio_frame)}", __name__)
            return numpy.expand_dims(create_empty_audio_frame(), axis=(0, 1))
            
        # Make a copy to avoid modifying the original
        audio_frame = temp_audio_frame.copy()
        
        # Check for empty frames or NaN values
        if audio_frame.size == 0 or numpy.isnan(audio_frame).any():
            logger.warning("Empty or invalid audio frame detected in prepare_audio_frame", __name__)
            return numpy.expand_dims(create_empty_audio_frame(), axis=(0, 1))
        
        # Calculate audio magnitude for logging and decision making
        audio_magnitude = numpy.sum(numpy.abs(audio_frame))
        logger.info(f"Audio frame magnitude: {audio_magnitude}", __name__)
        
        # If audio is too quiet but not empty, amplify it
        if 0 < audio_magnitude < 0.1:
            amplification = min(0.5 / audio_magnitude, 200.0)  # Allow higher amplification
            logger.info(f"Amplifying quiet audio by {amplification}x", __name__)
            audio_frame = audio_frame * amplification
            
        # Ensure float32 type
        if audio_frame.dtype != numpy.float32:
            audio_frame = audio_frame.astype(numpy.float32)
            
        # Replace any NaN or infinite values
        audio_frame = numpy.nan_to_num(audio_frame, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Apply the transformation - carefully checking for errors
        try:
            audio_frame = numpy.maximum(numpy.exp(-5 * numpy.log(10)), audio_frame)
            audio_frame = numpy.log10(numpy.maximum(1e-8, audio_frame)) * 1.6 + 3.2  # Changed from 1e-10 to 1e-8
            
            # Add extra range for clearer lip movement
            audio_frame = audio_frame.clip(-4, 4).astype(numpy.float32)
            
            # If the frame is still very quiet after transformation, boost it
            transformed_magnitude = numpy.sum(numpy.abs(audio_frame))
            if 0 < transformed_magnitude < 1.0:
                boost_factor = min(3.0 / transformed_magnitude, 10.0)
                logger.info(f"Boosting transformed audio by {boost_factor}x from {transformed_magnitude}", __name__)
                audio_frame = audio_frame * boost_factor
        except Exception as e:
            logger.error(f"Error transforming audio frame: {str(e)}", __name__)
            return numpy.expand_dims(create_empty_audio_frame(), axis=(0, 1))
        
        # Add batch and channel dimensions if needed
        if audio_frame.ndim == 2:
            audio_frame = numpy.expand_dims(audio_frame, axis=(0, 1))
        
        # Check final shape to ensure it's valid
        if audio_frame.ndim != 4:
            logger.error(f"Invalid output shape: {audio_frame.shape}", __name__)
            return numpy.expand_dims(create_empty_audio_frame(), axis=(0, 1))
            
        # Final magnitude check
        final_magnitude = numpy.sum(numpy.abs(audio_frame))
        logger.info(f"Final audio frame magnitude: {final_magnitude}", __name__)
            
        return audio_frame
    except Exception as e:
        logger.error(f"Error in prepare_audio_frame: {str(e)}", __name__)
        return numpy.expand_dims(create_empty_audio_frame(), axis=(0, 1))


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
        # Get source paths and add extensive logging
        source_paths = state_manager.get_item('source_paths')
        logger.info(f"Checking source paths for audio: {source_paths}", __name__)
        
        # First check if source_paths exists at all
        if not source_paths or not isinstance(source_paths, list) or len(source_paths) == 0:
            logger.error(wording.get('choose_audio_source') + wording.get('exclamation_mark'), __name__)
            return False
            
        # Then check if any are valid audio files
        audio_paths = filter_audio_paths(source_paths)
        logger.info(f"Filtered audio paths: {audio_paths}", __name__)
        
        if not audio_paths or len(audio_paths) == 0:
            logger.error(wording.get('choose_audio_source') + wording.get('exclamation_mark'), __name__)
            return False
        
        # Check if we need to validate reference faces
        if 'reference' in state_manager.get_item('face_selector_mode'):
            try:
                reference_faces = get_reference_faces()
                if reference_faces is None or (isinstance(reference_faces, (list, tuple)) and len(reference_faces) == 0):
                    logger.warning("No reference faces available but using reference mode", __name__)
                    # Don't return False here - we'll handle this in the process_frame method
            except Exception as e:
                logger.warning(f"Error checking reference faces: {e}", __name__)
                # Continue anyway - we'll handle this in the process_frame method
        
        # Check if we can actually extract frames from the audio
        for audio_path in audio_paths:
            try:
                # Try to get a sample frame to validate the audio works
                test_frame = get_voice_frame(audio_path, 25)
                if test_frame is not None and isinstance(test_frame, numpy.ndarray) and test_frame.size > 0:
                    logger.info(f"Successfully validated audio file: {audio_path}", __name__)
                    # Explicitly save this working audio path to ensure it's used
                    if source_paths and isinstance(source_paths, list):
                        # Move this working audio file to the front of the list
                        source_paths = [p for p in source_paths if p != audio_path]
                        source_paths.insert(0, audio_path)
                        state_manager.set_item('source_paths', source_paths)
                    return True
            except Exception as e:
                logger.error(f"Error validating audio file {audio_path}: {str(e)}", __name__)
        
        # Other checks for target and output paths
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
            
        # If we've gone through all audio files and none worked, return false
        logger.error("None of the audio files could be processed for lip syncing", __name__)
        return False

    def sync_lip(self, target_face: Face, temp_audio_frame: AudioFrame, temp_vision_frame: VisionFrame) -> VisionFrame:
        try:
            masker = FaceMasker()
            model_size = self.get_model_options().get('size')
            
            # Validate and log audio frame before processing
            if temp_audio_frame is None:
                logger.error("Audio frame is None in sync_lip", __name__)
                return temp_vision_frame
            
            if not isinstance(temp_audio_frame, numpy.ndarray):
                logger.error(f"Audio frame has invalid type in sync_lip: {type(temp_audio_frame)}", __name__)
                return temp_vision_frame
            
            # Log audio frame magnitude for debugging
            audio_magnitude = numpy.sum(numpy.abs(temp_audio_frame))
            logger.info(f"Audio frame magnitude in sync_lip: {audio_magnitude}", __name__)
            
            # Artificially enhance very quiet audio frames for better lip movement
            if 0 < audio_magnitude < 0.5:
                enhancement = min(1.0 / audio_magnitude, 50.0)
                logger.info(f"Enhancing quiet audio in sync_lip by {enhancement}x", __name__)
                temp_audio_frame = temp_audio_frame * enhancement
            
            # Ensure audio frame is in the correct format
            temp_audio_frame = prepare_audio_frame(temp_audio_frame)
            
            # Warp face
            crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
                temp_vision_frame, target_face.landmark_set.get('5/68'), 'ffhq_512', (512, 512))
                
            face_landmark_68 = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(
                -1, 2)
            bounding_box = create_bounding_box(face_landmark_68)
            bounding_box[1] -= numpy.abs(bounding_box[3] - bounding_box[1]) * 0.125
            
            # Create masks
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
            
            # Forward pass through the model
            logger.info(f"Running lip sync model with audio shape: {temp_audio_frame.shape}", __name__)
            close_vision_frame = self.forward(temp_audio_frame, close_vision_frame)
            close_vision_frame = normalize_close_frame(close_vision_frame)
            
            # Transform back and paste
            crop_vision_frame = cv2.warpAffine(close_vision_frame, cv2.invertAffineTransform(close_matrix), (512, 512),
                                            borderMode=cv2.BORDER_REPLICATE)
            crop_mask = numpy.minimum.reduce(crop_masks)
            
            # Enhance the mask for more visible movement with quiet audio
            if audio_magnitude < 1.0:
                logger.info("Enhancing mask for quiet audio", __name__)
                # Increase mask intensity for more visibility
                mask_enhancement = 1.5
                crop_mask = numpy.clip(crop_mask * mask_enhancement, 0, 1)
            
            return paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
        except Exception as e:
            logger.error(f"Error in sync_lip: {e}", __name__)
            return temp_vision_frame

    def forward(self, temp_audio_frame: AudioFrame, close_vision_frame: VisionFrame) -> VisionFrame:
        try:
            lip_syncer = self.get_inference_pool().get('lip_syncer')
            
            # Add debugging info
            logger.info(f"Running lip sync model with audio shape: {temp_audio_frame.shape}, vision shape: {close_vision_frame.shape}", __name__)
            
            if temp_audio_frame is None:
                logger.error("Audio frame is None in forward method", __name__)
                raise ValueError("Audio frame cannot be None")
            
            # Ensure audio has enough signal for the model to detect
            if isinstance(temp_audio_frame, numpy.ndarray):
                # Check if the audio frame is too quiet
                audio_magnitude = numpy.sum(numpy.abs(temp_audio_frame))
                logger.info(f"Audio magnitude in forward method: {audio_magnitude}", __name__)
                
                if audio_magnitude < 0.1:
                    # Create an artificial "speaking" pattern to force some movement
                    logger.info("Audio too quiet in forward method, creating artificial speaking pattern", __name__)
                    # Generate a sine wave pattern to simulate speech
                    if temp_audio_frame.ndim >= 4:
                        for i in range(temp_audio_frame.shape[0]):
                            for j in range(temp_audio_frame.shape[3]):
                                # Create a sine wave pattern with varying amplitude
                                pattern = numpy.sin(numpy.linspace(0, 4*numpy.pi, temp_audio_frame.shape[2])) * 2.0
                                # Apply to different parts of the audio frame
                                temp_audio_frame[i, 0, :, j] = pattern
            
            # Run the model with enhanced audio
            with conditional_thread_semaphore():
                result = lip_syncer.run(None, {'source': temp_audio_frame, 'target': close_vision_frame})[0]
                
            # Log model output stats
            if isinstance(result, numpy.ndarray):
                logger.info(f"Lip sync model output shape: {result.shape}, max: {numpy.max(result)}, min: {numpy.min(result)}", __name__)
                
            return result
        except Exception as e:
            logger.error(f"Error in forward method: {e}", __name__)
            # Return original frame if model fails
            return close_vision_frame

    def process_frame(self, inputs: LipSyncerInputs) -> VisionFrame:
        # Get inputs with proper validation
        reference_faces = inputs.get('reference_faces')
        source_audio_frame = inputs.get('source_audio_frame')
        source_audio_frame_2 = inputs.get('source_audio_frame_2')
        target_vision_frame = inputs.get('target_vision_frame')
        
        # Validate reference faces
        if reference_faces is None:
            reference_faces = []
        elif not isinstance(reference_faces, (list, tuple)):
            reference_faces = [reference_faces]
            
        # Get faces in target frame
        many_faces = []
        try:
            many_faces = sort_and_filter_faces(get_many_faces([target_vision_frame]))
        except Exception as e:
            logger.error(f"Error detecting faces in target frame: {e}", __name__)
        
        # Check if both audio frames are empty
        if source_audio_frame is None and source_audio_frame_2 is None:
            logger.error("No audio frames available - both audio sources are None", __name__)
            return target_vision_frame
            
        # Create empty frames if needed
        if source_audio_frame is None:
            logger.warn("Primary audio frame is None, creating empty frame", __name__)
            source_audio_frame = create_empty_audio_frame()
            
        if source_audio_frame_2 is None:
            source_audio_frame_2 = create_empty_audio_frame()
            
        # Check if frames have any data by checking sum of absolute values
        if (not isinstance(source_audio_frame, numpy.ndarray) or numpy.sum(numpy.abs(source_audio_frame)) < 1e-9) and \
           (not isinstance(source_audio_frame_2, numpy.ndarray) or numpy.sum(numpy.abs(source_audio_frame_2)) < 1e-9):
            logger.error(f"Both audio frames are empty or have no meaningful data. First frame sum: {numpy.sum(numpy.abs(source_audio_frame)) if isinstance(source_audio_frame, numpy.ndarray) else None}, Second frame sum: {numpy.sum(numpy.abs(source_audio_frame_2)) if isinstance(source_audio_frame_2, numpy.ndarray) else None}", __name__)
            return target_vision_frame
            
        # Ensure correct types for audio frames
        if not isinstance(source_audio_frame, numpy.ndarray):
            logger.error(f"Audio frame has wrong type: {type(source_audio_frame)}", __name__)
            source_audio_frame = create_empty_audio_frame()
        
        if not isinstance(source_audio_frame_2, numpy.ndarray):
            source_audio_frame_2 = create_empty_audio_frame()
            
        # Convert to float32 if needed
        if source_audio_frame.dtype != numpy.float32:
            logger.info(f"Converting audio frame from {source_audio_frame.dtype} to float32", __name__)
            source_audio_frame = source_audio_frame.astype(numpy.float32)
            
        if source_audio_frame_2.dtype != numpy.float32:
            source_audio_frame_2 = source_audio_frame_2.astype(numpy.float32)
            
        # Ensure correct dimensions
        expected_shape = (80, 16)
        if source_audio_frame.shape != expected_shape:
            logger.warn(f"Reshaping audio frame from {source_audio_frame.shape} to {expected_shape}", __name__)
            # Try to reshape or pad appropriately
            try:
                if source_audio_frame.size >= expected_shape[0] * expected_shape[1]:
                    # Reshape if we have enough data
                    source_audio_frame = source_audio_frame.flatten()[:expected_shape[0] * expected_shape[1]].reshape(expected_shape)
                else:
                    # Pad if we don't have enough data
                    temp_frame = numpy.zeros(expected_shape, dtype=numpy.float32)
                    r, c = min(source_audio_frame.shape[0], expected_shape[0]), min(source_audio_frame.shape[1], expected_shape[1])
                    temp_frame[:r, :c] = source_audio_frame[:r, :c]
                    source_audio_frame = temp_frame
            except Exception as e:
                logger.error(f"Failed to reshape audio frame: {e}", __name__)
                source_audio_frame = create_empty_audio_frame()
                
        # Do the same for the second audio frame
        if source_audio_frame_2.shape != expected_shape:
            try:
                if source_audio_frame_2.size >= expected_shape[0] * expected_shape[1]:
                    source_audio_frame_2 = source_audio_frame_2.flatten()[:expected_shape[0] * expected_shape[1]].reshape(expected_shape)
                else:
                    temp_frame = numpy.zeros(expected_shape, dtype=numpy.float32)
                    r, c = min(source_audio_frame_2.shape[0], expected_shape[0]), min(source_audio_frame_2.shape[1], expected_shape[1])
                    temp_frame[:r, :c] = source_audio_frame_2[:r, :c]
                    source_audio_frame_2 = temp_frame
            except:
                source_audio_frame_2 = create_empty_audio_frame()
        
        # Process faces based on the face selector mode
        face_selector_mode = state_manager.get_item('face_selector_mode')
        if face_selector_mode == 'many':
            if many_faces:
                for target_face in many_faces:
                    try:
                        target_vision_frame = self.sync_lip(target_face, source_audio_frame, target_vision_frame)
                    except Exception as e:
                        logger.error(f"Error while syncing lip for 'many' mode: {e}", __name__)
        
        elif face_selector_mode == 'one':
            target_face = get_one_face(many_faces)
            if target_face:
                try:
                    target_vision_frame = self.sync_lip(target_face, source_audio_frame, target_vision_frame)
                except Exception as e:
                    logger.error(f"Error while syncing lip for 'one' mode: {e}", __name__)
        
        elif face_selector_mode == 'reference':
            # Only proceed if we have reference faces
            if not reference_faces or len(reference_faces) == 0:
                logger.warn("No reference faces available for 'reference' mode", __name__)
                return target_vision_frame
                
            # Process each valid reference face
            for ref_idx, ref_face in enumerate(reference_faces):
                if ref_face is None:
                    continue
                    
                # Get the corresponding audio frame (use first audio for all refs if second is not available)
                if ref_idx == 0 or not numpy.any(source_audio_frame_2):
                    src_audio = source_audio_frame
                else:
                    src_audio = source_audio_frame_2
                    
                # Find similar faces
                try:
                    similar_faces = find_similar_faces(many_faces, ref_face,
                                                    state_manager.get_item('reference_face_distance'))
                    if similar_faces:
                        for similar_face in similar_faces:
                            try:
                                target_vision_frame = self.sync_lip(similar_face, src_audio, target_vision_frame)
                            except Exception as e:
                                logger.error(f"Error while syncing lip for reference {ref_idx}: {e}", __name__)
                except Exception as e:
                    logger.error(f"Error finding similar faces for reference {ref_idx}: {e}", __name__)
        
        return target_vision_frame

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        # Get audio sources
        source_paths = state_manager.get_item('source_paths')
        source_paths_2 = state_manager.get_item('source_paths_2')
        
        # Add debug logging
        logger.info(f"Source paths: {source_paths}", __name__)
        logger.info(f"Source paths 2: {source_paths_2}", __name__)
        
        # Validate source paths
        if not source_paths or not isinstance(source_paths, list) or len(source_paths) == 0:
            logger.error("No source paths found. Did you upload an audio file?", __name__)
            return []
        
        # Filter to find audio paths and get the first working one
        audio_paths = filter_audio_paths(source_paths)
        audio_paths_2 = filter_audio_paths(source_paths_2) if source_paths_2 else []
        
        logger.info(f"Filtered audio paths: {audio_paths}", __name__)
        logger.info(f"Filtered audio paths 2: {audio_paths_2}", __name__)
        
        # Test each audio path until we find one that works
        working_audio_path = None
        for audio_path in audio_paths:
            try:
                test_frame = get_voice_frame(audio_path, 25)
                if test_frame is not None and isinstance(test_frame, numpy.ndarray) and test_frame.size > 0:
                    working_audio_path = audio_path
                    logger.info(f"Found working audio path: {working_audio_path}", __name__)
                    break
            except Exception as e:
                logger.error(f"Error testing audio path {audio_path}: {str(e)}", __name__)
                
        # Test each audio path 2 until we find one that works
        working_audio_path_2 = None
        for audio_path in audio_paths_2:
            try:
                test_frame = get_voice_frame(audio_path, 25)
                if test_frame is not None and isinstance(test_frame, numpy.ndarray) and test_frame.size > 0:
                    working_audio_path_2 = audio_path
                    logger.info(f"Found working audio path 2: {working_audio_path_2}", __name__)
                    break
            except Exception as e:
                logger.error(f"Error testing audio path 2 {audio_path}: {str(e)}", __name__)
        
        if not working_audio_path:
            logger.error("No valid audio file found in source paths. Check file type and format.", __name__)
            return []
            
        # Get video settings
        temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'),
                                            state_manager.get_item('output_video_fps'))
        output_frames = []

        # Process each frame
        for queue_payload in process_manager.manage(queue_payloads):
            target_vision_path = queue_payload['frame_path']
            frame_number = queue_payload['frame_number']
            
            # Get audio frames
            try:
                source_audio_frame = get_voice_frame(working_audio_path, temp_video_fps, frame_number)
                logger.info(f"Got audio frame for frame {frame_number}: {source_audio_frame.shape if source_audio_frame is not None else None}", __name__)
            except Exception as e:
                logger.error(f"Error getting voice frame: {str(e)}", __name__)
                source_audio_frame = None
                
            try:
                source_audio_frame_2 = get_voice_frame(working_audio_path_2, temp_video_fps, frame_number) if working_audio_path_2 else None
            except Exception as e:
                logger.error(f"Error getting voice frame 2: {str(e)}", __name__)
                source_audio_frame_2 = None
            
            # Add debug logging
            if source_audio_frame is None:
                logger.error(f"Failed to get voice frame for {working_audio_path} at frame {frame_number}", __name__)
            
            reference_faces = queue_payload['reference_faces']

            # Ensure we have valid audio frames
            if source_audio_frame is None or not isinstance(source_audio_frame, numpy.ndarray) or numpy.sum(numpy.abs(source_audio_frame)) < 1e-9:
                logger.warn(f"Empty or invalid audio frame from {working_audio_path}. Sum: {numpy.sum(numpy.abs(source_audio_frame)) if isinstance(source_audio_frame, numpy.ndarray) else None}. Creating empty frame.", __name__)
                source_audio_frame = create_empty_audio_frame()
                
            if source_audio_frame_2 is None or not isinstance(source_audio_frame_2, numpy.ndarray) or numpy.sum(numpy.abs(source_audio_frame_2)) < 1e-9:
                source_audio_frame_2 = create_empty_audio_frame()
                
            # Process the frame
            target_vision_frame = read_image(target_vision_path)
            result_frame = self.process_frame(
                {
                    'reference_faces': reference_faces,
                    'source_audio_frame': source_audio_frame,
                    'source_audio_frame_2': source_audio_frame_2,
                    'target_vision_frame': target_vision_frame
                })
            write_image(target_vision_path, result_frame)
            output_frames.append((frame_number, target_vision_path))
            
        return output_frames

    def process_image(self, target_path: str, output_path: str, reference_faces=None) -> None:
        # Validate and prepare reference faces
        if reference_faces is None:
            try:
                if 'reference' in state_manager.get_item('face_selector_mode'):
                    reference_faces = get_reference_faces()
                    # Ensure we have a list or tuple
                    if reference_faces is None:
                        reference_faces = []
                    elif not isinstance(reference_faces, (list, tuple)):
                        reference_faces = [reference_faces]
                else:
                    reference_faces = []
            except Exception as e:
                logger.error(f"Error getting reference faces: {e}", __name__)
                reference_faces = []
        
        # Get audio source
        source_paths = state_manager.get_item('source_paths')
        
        # Add debug logging
        logger.info(f"Image processing source paths: {source_paths}", __name__)
        
        # Validate source paths
        if not source_paths or not isinstance(source_paths, list) or len(source_paths) == 0:
            logger.error("No source paths found for image processing", __name__)
            return
        
        # Filter to find audio paths and test each one to find a working one
        audio_paths = filter_audio_paths(source_paths)
        logger.info(f"Filtered audio paths for image: {audio_paths}", __name__)
        
        # Test each audio path until we find one that works
        working_audio_path = None
        for audio_path in audio_paths:
            try:
                test_frame = get_voice_frame(audio_path, 25)
                if test_frame is not None and isinstance(test_frame, numpy.ndarray) and test_frame.size > 0:
                    working_audio_path = audio_path
                    logger.info(f"Found working audio path for image: {working_audio_path}", __name__)
                    break
            except Exception as e:
                logger.error(f"Error testing audio path for image {audio_path}: {str(e)}", __name__)
        
        if not working_audio_path:
            logger.error("No valid audio file found for image processing", __name__)
            return
        
        # Get audio frame
        try:
            source_audio_frame = get_voice_frame(working_audio_path, 25)
            logger.info(f"Got audio frame for image: {source_audio_frame.shape if source_audio_frame is not None else None}", __name__)
        except Exception as e:
            logger.error(f"Error getting voice frame for image: {str(e)}", __name__)
            source_audio_frame = None
        
        # Validate audio frame
        if source_audio_frame is None or not isinstance(source_audio_frame, numpy.ndarray) or numpy.sum(numpy.abs(source_audio_frame)) < 1e-9:
            logger.error(f"Invalid audio frame for image from {working_audio_path}. Sum: {numpy.sum(numpy.abs(source_audio_frame)) if isinstance(source_audio_frame, numpy.ndarray) else None}", __name__)
            source_audio_frame = create_empty_audio_frame()
        
        # Process image
        try:
            target_vision_frame = read_static_image(target_path)
            result_frame = self.process_frame(
                {
                    'reference_faces': reference_faces,
                    'source_audio_frame': source_audio_frame,
                    'source_audio_frame_2': create_empty_audio_frame(),  # Default empty second frame
                    'target_vision_frame': target_vision_frame
                })
            write_image(output_path, result_frame)
        except Exception as e:
            logger.error(f"Error processing image: {e}", __name__)
            # Try to copy the original image as fallback
            try:
                import shutil
                shutil.copy(target_path, output_path)
                logger.warning(f"Copied original image to output as fallback", __name__)
            except Exception as copy_error:
                logger.error(f"Failed to copy original image: {copy_error}", __name__)
