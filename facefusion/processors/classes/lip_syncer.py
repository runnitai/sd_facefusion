import os
import traceback
from argparse import ArgumentParser
from typing import List, Tuple

import cv2
import numpy
import numpy as np
import torch

from facefusion import logger, state_manager, wording
from facefusion.download import conditional_download_sources_no_hash
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.face_helper import create_bounding_box, paste_back, warp_face_by_face_landmark_5
from facefusion.face_selector import find_similar_faces, sort_and_filter_faces
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import filter_audio_paths, in_directory, is_image, is_video, \
    resolve_relative_path, same_file_extension, has_audio
from facefusion.musetalk.utils.audio_processor import AudioProcessor
from facefusion.musetalk.utils.blending import get_image_prepare_material, get_image_blending
from facefusion.musetalk.utils.face_parsing import FaceParsing
from facefusion.musetalk.utils.utils import load_all_model, datagen
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.typing import LipSyncerInputs
from facefusion.program_helper import find_argument_group
from facefusion.typing import ApplyStateItem, Args, Face, InferencePool, ModelSet, \
    ProcessMode, QueuePayload, VisionFrame
from facefusion.vision import read_image, read_static_image, restrict_video_fps, write_image


class LipSyncer(BaseProcessor):
    MODEL_SET: ModelSet = {
        'musetalk_v15': {
            'hashes': {
                # No hash validation for HuggingFace models
            },
            'sources': {
                'musetalk_unet': {
                    'url': 'https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth?download=true',
                    'path': resolve_relative_path('../.assets/models/musetalk_v15/unet.pth')
                },
                'musetalk_config': {
                    'url': 'https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json?download=true',
                    'path': resolve_relative_path('../.assets/models/musetalk_v15/musetalk.json')
                },
                'sd_vae_config': {
                    'url': 'https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json?download=true',
                    'path': resolve_relative_path('../.assets/models/musetalk_v15/sd-vae-ft-mse/config.json')
                },
                'sd_vae_model': {
                    'url': 'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors?download=true',
                    'path': resolve_relative_path(
                        '../.assets/models/musetalk_v15/sd-vae-ft-mse/vae-ft-mse-840000-ema-pruned.safetensors')
                },
                'whisper_config': {
                    'url': 'https://huggingface.co/openai/whisper-tiny/resolve/main/config.json?download=true',
                    'path': resolve_relative_path('../.assets/models/musetalk_v15/whisper-tiny/config.json')
                },
                'whisper_preprocessor_config': {
                    'url': 'https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json?download=true',
                    'path': resolve_relative_path(
                        '../.assets/models/musetalk_v15/whisper-tiny/preprocessor_config.json')
                },
                'whisper_model': {
                    'url': 'https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors?download=true',
                    'path': resolve_relative_path('../.assets/models/musetalk_v15/whisper-tiny/pytorch_model.bin')
                },
                'whisper_tokenizer': {
                    'url': 'https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json?download=true',
                    'path': resolve_relative_path('../.assets/models/musetalk_v15/whisper-tiny/tokenizer.json')
                },
                'whisper_vocab': {
                    'url': 'https://huggingface.co/openai/whisper-tiny/resolve/main/vocab.json?download=true',
                    'path': resolve_relative_path('../.assets/models/musetalk_v15/whisper-tiny/vocab.json')
                },
                'face_parse_resnet': {
                    'url': 'https://github.com/fregu856/deeplabv3/raw/refs/heads/master/pretrained_models/resnet/resnet18-5c106cde.pth',
                    'path': resolve_relative_path(
                        '../.assets/models/musetalk_v15/face-parse-bisent/resnet18-5c106cde.pth')
                },
                'face_parse_bisent': {
                    'url': 'https://github.com/zllrunning/face-makeup.PyTorch/raw/refs/heads/master/cp/79999_iter.pth',
                    'path': resolve_relative_path('../.assets/models/musetalk_v15/face-parse-bisent/79999_iter.pth')
                }
            },
            'size': (256, 256)
        }
    }

    model_key: str = 'musetalk_model'
    priority = 10

    # MuseTalk components
    _musetalk_vae = None
    _musetalk_unet = None
    _musetalk_pe = None
    _musetalk_face_parsing = None
    _musetalk_audio_processor = None
    _musetalk_whisper = None
    
    # Global audio chunks storage (like realtime_inference.py)
    _whisper_chunks = None
    _whisper_chunks_2 = None
    _current_audio_path = None
    _current_audio_path_2 = None
    _silent_chunks = None  # Set of silent chunk indices for source 1
    _silent_chunks_2 = None  # Set of silent chunk indices for source 2

    def register_args(self, program: ArgumentParser) -> None:
        group = find_argument_group(program, 'processors')
        if group:
            group.add_argument('--lip-sync-empty-audio', action='store_true', help='sync lip movements even when no audio is detected')

    def apply_args(self, args: Args, apply_state_item: ApplyStateItem) -> None:
        state_manager.set_item('lip_sync_empty_audio', args.get('lip_sync_empty_audio', False))

    def _initialize_musetalk(self):
        """Initialize MuseTalk components following original project structure"""
        if self._musetalk_vae is None:
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Load MuseTalk models using the original workflow
                self._musetalk_vae, self._musetalk_unet, self._musetalk_pe = load_all_model(device=device)

                # Ensure proper dtype
                weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self._musetalk_unet.model = self._musetalk_unet.model.to(dtype=weight_dtype)
                
                # Initialize audio processor exactly like original MuseTalk
                whisper_model_path = resolve_relative_path('../.assets/models/musetalk_v15/whisper-tiny')
                self._musetalk_audio_processor = AudioProcessor(feature_extractor_path=whisper_model_path)
                
                # Initialize Whisper for audio feature extraction
                from transformers import WhisperModel
                self._musetalk_whisper = WhisperModel.from_pretrained(whisper_model_path)
                self._musetalk_whisper.to(device)
                self._musetalk_whisper = self._musetalk_whisper.to(dtype=weight_dtype)
                
                # Initialize face parsing
                try:
                    self._musetalk_face_parsing = FaceParsing()
                    if self._musetalk_face_parsing is None or not hasattr(self._musetalk_face_parsing, 'net'):
                        logger.error("FaceParsing initialization returned None or invalid object", __name__)
                        raise RuntimeError("FaceParsing initialization failed")
                except Exception as fp_error:
                    logger.error(f"FaceParsing initialization failed: {fp_error}", __name__)
                    traceback.print_exc()
                    self._musetalk_face_parsing = None
                    raise fp_error

                logger.info("MuseTalk models initialized successfully", __name__)

            except Exception as e:
                logger.error(f"MuseTalk initialization failed: {e}", __name__)
                logger.error(f"Traceback: {traceback.format_exc()}", __name__)
                raise e

    def _get_whisper_chunks_with_silence_detection(self, audio_path: str, fps: float, weight_dtype, device):
        """Get whisper chunks and detect silence during processing"""
        try:
            import librosa
            import math
            from einops import rearrange
            
            # Load raw audio for silence detection
            raw_audio, sr = librosa.load(audio_path, sr=16000)
            
            # Get whisper features using the audio processor
            whisper_input_features, librosa_length = self._musetalk_audio_processor.get_audio_feature(audio_path, weight_dtype=weight_dtype)
            
            # Process whisper features
            audio_feature_length_per_frame = 2 * (2 + 2 + 1)  # audio_padding_length_left=2, right=2
            whisper_feature = []
            
            for input_feature in whisper_input_features:
                input_feature = input_feature.to(device).to(weight_dtype)
                audio_feats = self._musetalk_whisper.encoder(input_feature, output_hidden_states=True).hidden_states
                audio_feats = torch.stack(audio_feats, dim=2)
                whisper_feature.append(audio_feats)

            whisper_feature = torch.cat(whisper_feature, dim=1)
            
            # Trim and pad like original
            audio_fps = 50
            fps = int(fps)
            whisper_idx_multiplier = audio_fps / fps
            num_frames = math.floor((librosa_length / sr) * fps)
            actual_length = math.floor((librosa_length / sr) * audio_fps)
            whisper_feature = whisper_feature[:,:actual_length,...]

            padding_nums = math.ceil(whisper_idx_multiplier)
            whisper_feature = torch.cat([
                torch.zeros_like(whisper_feature[:, :padding_nums * 2]),  # left padding
                whisper_feature,
                torch.zeros_like(whisper_feature[:, :padding_nums * 6])   # right padding
            ], 1)

            # Process chunks and detect silence simultaneously
            audio_prompts = []
            silent_chunks = set()
            
            for frame_index in range(num_frames):
                # Get whisper chunk
                audio_index = math.floor(frame_index * whisper_idx_multiplier)
                audio_clip = whisper_feature[:, audio_index: audio_index + audio_feature_length_per_frame]
                audio_clip = rearrange(audio_clip, 'b c h w -> b (c h) w')
                audio_prompts.append(audio_clip)
                
                # Check if corresponding raw audio segment is silent
                frame_duration = 1.0 / fps
                start_sample = int(frame_index * frame_duration * sr)
                end_sample = int((frame_index + 1) * frame_duration * sr)
                
                if start_sample < len(raw_audio):
                    audio_segment = raw_audio[start_sample:min(end_sample, len(raw_audio))]
                    
                    # Use librosa.trim to detect if there's meaningful audio
                    trimmed, _ = librosa.effects.trim(audio_segment, top_db=30)  # 30dB threshold
                    
                    # Consider silent if trimmed audio is very short or has low energy
                    is_silent = (len(trimmed) < len(audio_segment) * 0.1 or 
                               np.max(np.abs(trimmed)) < 0.01)
                    
                    if is_silent:
                        silent_chunks.add(frame_index)
                else:
                    # Beyond audio length
                    silent_chunks.add(frame_index)

            audio_prompts = torch.cat(audio_prompts, dim=0)
            
            print(f"Processed {len(audio_prompts)} chunks, {len(silent_chunks)} detected as silent")
            return audio_prompts, silent_chunks
            
        except Exception as e:
            print(f"Error in whisper chunk processing with silence detection: {e}")
            import traceback
            traceback.print_exc()
            return [], set()

    def _process_and_cache_audio(self, audio_path: str, fps: float, audio_source: int = 1):
        """Process audio upfront and cache chunks like realtime_inference.py"""
        try:
            # Choose the right cache variables based on audio source
            if audio_source == 1:
                current_path = self._current_audio_path
                cached_chunks = self._whisper_chunks
                cached_silent = self._silent_chunks
            else:
                current_path = self._current_audio_path_2
                cached_chunks = self._whisper_chunks_2
                cached_silent = self._silent_chunks_2
            
            # Skip if already processed
            if current_path == audio_path and cached_chunks is not None:
                print(f"Using cached audio chunks for source {audio_source}: {audio_path} ({len(cached_chunks)} chunks, {len(cached_silent) if cached_silent else 0} silent)")
                return cached_chunks
                
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            print(f"Processing and caching audio source {audio_source}: {audio_path}")
            
            # Process whisper chunks with integrated silence detection
            whisper_chunks, silent_chunks = self._get_whisper_chunks_with_silence_detection(audio_path, fps, weight_dtype, device)
            
            # Cache the results in the appropriate variables
            if audio_source == 1:
                self._whisper_chunks = whisper_chunks
                self._current_audio_path = audio_path
                self._silent_chunks = silent_chunks
            else:
                self._whisper_chunks_2 = whisper_chunks
                self._current_audio_path_2 = audio_path
                self._silent_chunks_2 = silent_chunks
            
            print(f"Cached {len(whisper_chunks)} audio chunks for source {audio_source}: {audio_path} ({len(silent_chunks)} silent)")
            return whisper_chunks

        except Exception as e:
            print(f"Audio processing failed for source {audio_source}: {e}")
            traceback.print_exc()
            return []

    def get_audio_chunk_for_frame(self, frame_index: int, audio_source: int = 1) -> tuple:
        """Get the appropriate audio chunk for a specific frame index and whether it's silent"""
        try:
            # Handle negative frame indices (use 0 instead)
            if frame_index < 0:
                frame_index = 0
                
            # Choose the right audio chunks and silent chunks
            chunks = self._whisper_chunks if audio_source == 1 else self._whisper_chunks_2
            silent_chunks = self._silent_chunks if audio_source == 1 else self._silent_chunks_2
            
            # Properly check if chunks is None or empty without tensor boolean evaluation
            if chunks is None:
                print(f"No audio chunks available for source {audio_source} (chunks is None)")
                return None, False
            
            # Check if chunks is a list/sequence and if it's empty
            try:
                chunks_len = len(chunks)
                if chunks_len == 0:
                    print(f"No audio chunks available for source {audio_source} (empty list)")
                    return None, False
            except TypeError:
                # If chunks doesn't support len(), it's probably not a valid chunks list
                print(f"Invalid chunks type for source {audio_source}: {type(chunks)}")
                return None, False
                
            # Use modulo to cycle through chunks like datagen does
            chunk_index = frame_index % chunks_len
            audio_chunk = chunks[chunk_index]
            
            # Check if this chunk is silent
            is_silent = silent_chunks is not None and chunk_index in silent_chunks
            
            # Ensure we return a proper tensor
            if isinstance(audio_chunk, torch.Tensor):
                #print(f"Retrieved audio chunk {chunk_index} for frame {frame_index} from source {audio_source}, shape: {audio_chunk.shape}, silent: {is_silent}")
                return audio_chunk, is_silent
            else:
                #print(f"Audio chunk is not a tensor: {type(audio_chunk)}")
                return None, False
            
        except Exception as e:
            print(f"Error getting audio chunk for frame {frame_index}: {e}")
            import traceback
            traceback.print_exc()
            return None, False

    def pre_check(self) -> bool:
        """Download MuseTalk models if needed"""
        download_directory_path = resolve_relative_path(self.model_path)
        model_sources = self.get_model_options().get('sources', {})

        all_downloaded = True
        for model_key, model_info in model_sources.items():
            model_path = model_info.get('path')

            # Create the directory for this specific model
            model_dir = os.path.dirname(model_path)
            os.makedirs(model_dir, exist_ok=True)

            # Create a single-item source dict for this model
            single_source = {model_key: model_info}

            downloaded = conditional_download_sources_no_hash(model_dir, single_source)

            if downloaded:
                logger.info(f"✓ {model_key} downloaded successfully", __name__)
            else:
                logger.error(f"✗ Failed to download {model_key}", __name__)
                all_downloaded = False

        return all_downloaded

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

        # Initialize MuseTalk and process audio upfront
        self._initialize_musetalk()
        
        # Get audio paths and process them
        source_paths = state_manager.get_item('source_paths')
        source_paths_2 = state_manager.get_item('source_paths_2')
        
        fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
        
        if source_paths:
            audio_paths = filter_audio_paths(source_paths)
            if audio_paths:
                self._process_and_cache_audio(audio_paths[0], fps, audio_source=1)
        
        if source_paths_2:
            audio_paths_2 = filter_audio_paths(source_paths_2)
            if audio_paths_2:
                self._process_and_cache_audio(audio_paths_2[0], fps, audio_source=2)

        return True

    def sync_lip(self, target_face: Face, audio_chunk: torch.Tensor, temp_vision_frame: VisionFrame, is_silent: bool = False) -> VisionFrame:
        """Main lip sync method using MuseTalk with properly processed audio chunk"""
        try:
            # Ensure MuseTalk models are initialized
            self._initialize_musetalk()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Check if we should sync empty audio
            sync_empty_audio = state_manager.get_item('lip_sync_empty_audio')
            if sync_empty_audio is None:
                sync_empty_audio = False
            
            # Handle silent audio chunks based on raw audio analysis
            if audio_chunk is None or is_silent:
                if not sync_empty_audio:
                    # Return original frame without modification
                    print(f"Silent audio chunk detected and sync_empty_audio is False - returning original frame")
                    return temp_vision_frame
                else:
                    # Use small random features for natural mouth movement
                    audio_chunk = torch.randn(1, 50, 384, device=device, dtype=weight_dtype) * 0.05
                    print(f"Silent audio chunk detected but sync_empty_audio is True - using random features")
            
            # Face processing - align to 512x512 as per MuseTalk requirements
            crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame,
                                                                            target_face.landmark_set.get('5/68'),
                                                                            'ffhq_512', (512, 512))
            face_landmark_68 = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2),
                                             affine_matrix).reshape(-1, 2)
            bounding_box = create_bounding_box(face_landmark_68)

            # Adjust bounding box for MuseTalk face region (256x256)
            x1, y1, x2, y2 = bounding_box
            extra_margin = 10
            y2 = min(y2 + extra_margin, crop_vision_frame.shape[0])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            face_crop = crop_vision_frame[y1:y2, x1:x2]
            face_crop_resized = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # Process audio chunk through PE like realtime_inference.py
            # Add batch dimension if missing (PE expects [batch, seq_len, d_model])
            if audio_chunk.dim() == 2:
                audio_chunk = audio_chunk.unsqueeze(0)  # Add batch dimension: [50, 384] -> [1, 50, 384]
            
            audio_features = self._musetalk_pe(audio_chunk.to(device))
            
            # MuseTalk inference - single step latent space inpainting
            latents = self._musetalk_vae.get_latents_for_unet(face_crop_resized)
            latents = latents.to(dtype=weight_dtype, device=device)
            audio_features = audio_features.to(dtype=weight_dtype, device=device)

            timesteps = torch.tensor([0], device=device, dtype=weight_dtype)

            # Single-step UNet inference (NOT diffusion - this is key!)
            with torch.no_grad():
                pred_latents = self._musetalk_unet.model(latents, timesteps,
                                                         encoder_hidden_states=audio_features).sample

            # Decode latents back to image
            result_frame = self._musetalk_vae.decode_latents(pred_latents)
            result_frame = result_frame[0]

            # Resize back to original face size
            result_frame_resized = cv2.resize(result_frame.astype(numpy.uint8), (x2 - x1, y2 - y1))
            
            # Apply face blending if face parsing is available
            try:
                if self._musetalk_face_parsing is not None:
                    mask_array, crop_box = get_image_prepare_material(
                        crop_vision_frame,
                        [x1, y1, x2, y2],
                        fp=self._musetalk_face_parsing,
                        mode="raw"
                    )

                    blended_frame = get_image_blending(
                        crop_vision_frame,
                        result_frame_resized,
                        [x1, y1, x2, y2],
                        mask_array,
                        crop_box
                    )
                else:
                    # Fallback to simple paste
                    blended_frame = crop_vision_frame.copy()
                    blended_frame[y1:y2, x1:x2] = result_frame_resized

                # Paste back to original frame
                paste_vision_frame = paste_back(temp_vision_frame, blended_frame,
                                                numpy.ones_like(blended_frame[:, :, 0]), affine_matrix)

            except Exception as e:
                logger.warn(f"Face blending failed, using simple paste: {e}", __name__)
                crop_vision_frame[y1:y2, x1:x2] = result_frame_resized
                paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame,
                                                numpy.ones_like(crop_vision_frame[:, :, 0]), affine_matrix)

            return paste_vision_frame

        except Exception as e:
            logger.error(f"MuseTalk sync failed: {e}", __name__)
            logger.error(f"Traceback: {traceback.format_exc()}", __name__)
            return temp_vision_frame

    def get_inference_pool(self) -> InferencePool:
        """Initialize and return MuseTalk inference pool"""
        self._initialize_musetalk()
        return {
            'musetalk_vae': self._musetalk_vae,
            'musetalk_unet': self._musetalk_unet,
            'musetalk_pe': self._musetalk_pe,
            'musetalk_face_parsing': self._musetalk_face_parsing,
            'musetalk_audio_processor': self._musetalk_audio_processor,
            'musetalk_whisper': self._musetalk_whisper
        }

    def clear_inference_pool(self) -> None:
        """Clear MuseTalk models from memory"""
        if hasattr(self, '_musetalk_vae') and self._musetalk_vae is not None:
            del self._musetalk_vae
            del self._musetalk_unet
            del self._musetalk_pe
            del self._musetalk_face_parsing
            del self._musetalk_audio_processor
            del self._musetalk_whisper

            self._musetalk_vae = None
            self._musetalk_unet = None
            self._musetalk_pe = None
            self._musetalk_face_parsing = None
            self._musetalk_audio_processor = None
            self._musetalk_whisper = None

    def process_frame(self, inputs: LipSyncerInputs) -> VisionFrame:
        reference_faces = inputs.get('reference_faces')
        # Get frame index if available, otherwise use 0
        frame_index = inputs.get('frame_index', 0)
        target_vision_frame = inputs.get('target_vision_frame')

        many_faces = []
        try:
            many_faces = sort_and_filter_faces(get_many_faces([target_vision_frame]))
        except Exception as e:
            logger.error(f"Error detecting faces in target frame: {e}", __name__)

        face_selector_mode = state_manager.get_item('face_selector_mode')

        if face_selector_mode == 'many':
            if many_faces:
                for i, target_face in enumerate(many_faces):
                    try:
                        audio_chunk, is_silent = self.get_audio_chunk_for_frame(frame_index, 1)
                        target_vision_frame = self.sync_lip(target_face, audio_chunk, target_vision_frame, is_silent)
                    except Exception as e:
                        logger.error(f"Error while syncing lip for face {i + 1} in 'many' mode: {e}", __name__)

        elif face_selector_mode == 'one':
            target_face = get_one_face(many_faces)
            if target_face:
                try:
                    audio_chunk, is_silent = self.get_audio_chunk_for_frame(frame_index, 1)
                    target_vision_frame = self.sync_lip(target_face, audio_chunk, target_vision_frame, is_silent)
                except Exception as e:
                    logger.error(f"Error while syncing lip for 'one' mode: {e}", __name__)

        elif face_selector_mode == 'reference':
            if not reference_faces or len(reference_faces) == 0:
                logger.warn("No reference faces available for 'reference' mode", __name__)
                return target_vision_frame

            # Process each valid reference face
            for ref_idx, ref_face in reference_faces.items():
                if ref_face is None:
                    continue

                # Get the corresponding audio chunk
                audio_chunk, is_silent = self.get_audio_chunk_for_frame(frame_index, 1 if ref_idx == 0 else 2)

                # Find similar faces
                try:
                    similar_faces = find_similar_faces(many_faces, ref_face,
                                                       state_manager.get_item('reference_face_distance'))

                    if similar_faces:
                        for sim_idx, similar_face in enumerate(similar_faces):
                            try:
                                target_vision_frame = self.sync_lip(similar_face, audio_chunk, target_vision_frame, is_silent)
                            except Exception as e:
                                logger.error(
                                    f"Error while syncing lip for similar face {sim_idx + 1} of reference {ref_idx + 1}: {e}",
                                    __name__)
                except Exception as e:
                    logger.error(f"Error finding similar faces for reference {ref_idx + 1}: {e}", __name__)

        return target_vision_frame

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        """Process frames using the cached audio chunks - simple frame-by-frame processing"""
        
        # Ensure models are initialized and audio is processed
        self._initialize_musetalk()
        
        # Properly check if whisper_chunks is None or empty without tensor boolean evaluation
        if self._whisper_chunks is None:
            logger.error("No audio chunks available (whisper_chunks is None). Did you upload an audio file?", __name__)
            return []
        
        try:
            chunks_len = len(self._whisper_chunks)
            if chunks_len == 0:
                logger.error("No audio chunks available (empty chunks). Did you upload an audio file?", __name__)
                return []
        except TypeError:
            logger.error("Invalid whisper_chunks type. Did you upload an audio file?", __name__)
            return []

        # Simple frame-by-frame processing - much more memory efficient
        output_frames = []
        
        for queue_payload in queue_payloads:
            target_vision_path = queue_payload['frame_path']
            frame_number = queue_payload['frame_number']
            
            try:
                # Read the frame
                target_vision_frame = read_image(target_vision_path)
                
                # Get reference faces if needed
                reference_faces = get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else None
                
                # Process the frame using the same logic as preview
                result_frame = self.process_frame({
                    'reference_faces': reference_faces,
                    'frame_index': frame_number,  # Use frame number for audio chunk selection
                    'target_vision_frame': target_vision_frame
                })
                
                # Write the processed frame back
                write_image(target_vision_path, result_frame)
                output_frames.append((frame_number, target_vision_path))
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_number}: {e}", __name__)
                # On error, just add the original frame
                output_frames.append((frame_number, target_vision_path))

        return output_frames

    def process_image(self, target_path: str, output_path: str, reference_faces=None) -> None:
        if not reference_faces:
            reference_faces = get_reference_faces() if 'reference' in state_manager.get_item(
                'face_selector_mode') else None
        
        target_vision_frame = read_static_image(target_path)
        result_frame = self.process_frame(
            {
                'reference_faces': reference_faces,
                'frame_index': 0,  # For images, use first audio chunk
                'target_vision_frame': target_vision_frame
            })
        write_image(output_path, result_frame)
