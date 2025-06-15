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
from facefusion.ffmpeg import ensure_wav_audio
from facefusion.face_helper import create_bounding_box, paste_back, warp_face_by_face_landmark_5
from facefusion.face_selector import find_similar_faces, sort_and_filter_faces
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import filter_audio_paths, in_directory, is_image, is_video, \
    resolve_relative_path, same_file_extension, has_audio
from facefusion.musetalk.utils.audio_processor import AudioProcessor
from facefusion.musetalk.utils.face_parsing import FaceParsing
from facefusion.musetalk.utils.utils import load_all_model
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.typing import LipSyncerInputs
from facefusion.processors.optimizations.gpu_cv_ops import resize_gpu_or_cpu
from facefusion.program_helper import find_argument_group
from facefusion.typing import ApplyStateItem, Args, Face, InferencePool, ModelSet, \
    ProcessMode, QueuePayload, VisionFrame
from facefusion.vision import read_image, read_static_image, restrict_video_fps, write_image
from facefusion.workers.classes.face_masker import FaceMasker
from transformers import WhisperModel
import librosa
import math
from einops import rearrange


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
    _silent_template = None  # Template silent chunk for source 1
    _silent_template_2 = None  # Template silent chunk for source 2
    _pause_chunks = None  # Set of pause chunk indices for source 1
    _pause_chunks_2 = None  # Set of pause chunk indices for source 2
    _pause_template = None  # Template pause chunk for source 1
    _pause_template_2 = None  # Template pause chunk for source 2
    masker = None
    face_mask_padding = (0, 0, 0, 0)  # Default padding for face mask
    face_mask_types = ['region']  # Default mask type
    face_mask_regions = ['mouth']  # Default regions for lip sync
    enhanced_blur = 0.4  # Default blur for lip sync mask
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def register_args(self, program: ArgumentParser) -> None:
        group = find_argument_group(program, 'processors')
        if group:
            group.add_argument('--lip-sync-keep-audio', action='store_true',
                               help='Use the original audio when creating the final video')

    def apply_args(self, args: Args, apply_state_item: ApplyStateItem) -> None:
        """Apply processor-specific CLI or job arguments.

        This previously *always* reset the ``lip_sync_keep_audio`` flag to
        ``False`` when the argument was not explicitly provided in ``args``.
        As a consequence, a value that had been set in the Gradio UI via the
        *Keep Original Audio* checkbox was lost during batch / job execution
        because the key is not included in the job-step arguments.

        To preserve the flag that was already saved in ``state_manager`` we
        now only overwrite it **if** the argument is present in the incoming
        ``args`` mapping. This guarantees the following behaviour:

        1. UI run – checkbox state is honoured (unchanged).
        2. Headless / job run – the checkbox state captured at job creation
           is kept, unless the user explicitly passes the
           ``--lip-sync-keep-audio`` CLI argument.
        """

        if 'lip_sync_keep_audio' in args:
            # Use ApplyStateItem callback (set or init) if provided, otherwise
            # fall back to direct state_manager access.
            try:
                apply_state_item('lip_sync_keep_audio', args.get('lip_sync_keep_audio'))
            except Exception:
                state_manager.set_item('lip_sync_keep_audio', args.get('lip_sync_keep_audio'))
        else:
            state_manager.set_item('lip_sync_keep_audio', False)

    def _initialize_musetalk(self):
        """Initialize MuseTalk components following original project structure"""
        if self._musetalk_vae is None:
            try:
                # Load MuseTalk models using the original workflow
                self._musetalk_vae, self._musetalk_unet, self._musetalk_pe = load_all_model(device=self.device)

                # Ensure proper dtype
                self._musetalk_unet.model = self._musetalk_unet.model.to(dtype=self.weight_dtype)

                # Initialize audio processor exactly like original MuseTalk
                whisper_model_path = resolve_relative_path('../.assets/models/musetalk_v15/whisper-tiny')
                self._musetalk_audio_processor = AudioProcessor(feature_extractor_path=whisper_model_path)

                # Initialize Whisper for audio feature extraction
                self._musetalk_whisper = WhisperModel.from_pretrained(whisper_model_path)
                self._musetalk_whisper.to(self.device)
                self._musetalk_whisper = self._musetalk_whisper.to(dtype=self.weight_dtype)

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
        """Get whisper chunks with smart silence detection - handles pauses vs true silence"""
        try:
            # Load raw audio for silence detection - ensure consistent 16kHz sampling
            raw_audio, sr = librosa.load(audio_path, sr=16000)
            
            # Calculate precise timing parameters
            audio_fps = 50.0  # MuseTalk internal audio fps
            video_fps = float(fps)
            
            # Calculate exact frame timing for perfect sync
            frame_duration_seconds = 1.0 / video_fps
            samples_per_video_frame = int(16000 * frame_duration_seconds)
            
            # Get total duration from raw audio
            total_duration_seconds = len(raw_audio) / 16000.0
            total_video_frames = int(total_duration_seconds * video_fps)

            # Get whisper features using the audio processor
            whisper_input_features, librosa_length = self._musetalk_audio_processor.get_audio_feature(audio_path,
                                                                                                      weight_dtype=weight_dtype)

            # Process whisper features
            audio_feature_length_per_frame = 2 * (2 + 2 + 1)  # audio_padding_length_left=2, right=2
            whisper_feature = []

            for input_feature in whisper_input_features:
                input_feature = input_feature.to(device).to(weight_dtype)
                audio_feats = self._musetalk_whisper.encoder(input_feature, output_hidden_states=True).hidden_states
                audio_feats = torch.stack(audio_feats, dim=2)
                whisper_feature.append(audio_feats)

            whisper_feature = torch.cat(whisper_feature, dim=1)

            # Calculate precise whisper frame indexing
            whisper_idx_multiplier = audio_fps / video_fps
            actual_audio_length = math.floor((librosa_length / 16000.0) * audio_fps)
            whisper_feature = whisper_feature[:, :actual_audio_length, ...]

            # Add padding with precise calculations
            padding_nums = math.ceil(whisper_idx_multiplier)
            whisper_feature = torch.cat([
                torch.zeros_like(whisper_feature[:, :padding_nums * 2]),  # left padding
                whisper_feature,
                torch.zeros_like(whisper_feature[:, :padding_nums * 6])  # right padding
            ], 1)

            # Smart silence detection parameters
            pause_buffer_frames = max(2, int(video_fps * 0.15))  # 150ms buffer for short pauses
            true_silence_frames = max(5, int(video_fps * 0.5))   # 500ms for true silence
            
            # Thresholds for different silence types
            pause_rms_threshold = 0.015    # Higher threshold for pauses (less aggressive)
            pause_peak_threshold = 0.08    # Allow some energy during pauses
            silence_rms_threshold = 0.008  # Lower threshold for true silence
            silence_peak_threshold = 0.03  # Strict threshold for true silence

            # Process chunks with smart silence detection
            audio_prompts = []
            raw_energy_values = []  # Track energy for temporal analysis
            
            # First pass: extract audio prompts and calculate energy
            for frame_index in range(total_video_frames):
                # Calculate precise audio index for this video frame
                precise_audio_time = frame_index * frame_duration_seconds
                audio_index = int(precise_audio_time * audio_fps)
                
                # Get whisper chunk with exact timing
                audio_clip = whisper_feature[:, audio_index: audio_index + audio_feature_length_per_frame]
                audio_clip = rearrange(audio_clip, 'b c h w -> b (c h) w')
                audio_prompts.append(audio_clip)

                # Calculate energy for this frame
                start_sample = int(frame_index * samples_per_video_frame)
                end_sample = int((frame_index + 1) * samples_per_video_frame)

                if start_sample < len(raw_audio):
                    audio_segment = raw_audio[start_sample:min(end_sample, len(raw_audio))]
                    if len(audio_segment) > 0:
                        rms_energy = np.sqrt(np.mean(audio_segment**2))
                        peak_energy = np.max(np.abs(audio_segment))
                        raw_energy_values.append((rms_energy, peak_energy))
                    else:
                        raw_energy_values.append((0.0, 0.0))
                else:
                    raw_energy_values.append((0.0, 0.0))

            # Second pass: smart silence classification with temporal context
            silence_states = []  # 0=speaking, 1=pause, 2=true_silence
            
            for frame_index in range(total_video_frames):
                rms_energy, peak_energy = raw_energy_values[frame_index]
                
                # Check if current frame is quiet
                is_quiet = (rms_energy < pause_rms_threshold and peak_energy < pause_peak_threshold)
                is_very_quiet = (rms_energy < silence_rms_threshold and peak_energy < silence_peak_threshold)
                
                if not is_quiet:
                    # Definitely speaking
                    silence_states.append(0)
                elif is_very_quiet:
                    # Check temporal context for true silence
                    # Look ahead and behind to see if this is extended silence
                    start_check = max(0, frame_index - true_silence_frames//2)
                    end_check = min(total_video_frames, frame_index + true_silence_frames//2)
                    
                    context_quiet_count = 0
                    for check_idx in range(start_check, end_check):
                        check_rms, check_peak = raw_energy_values[check_idx]
                        if check_rms < silence_rms_threshold and check_peak < silence_peak_threshold:
                            context_quiet_count += 1
                    
                    # If most of the context is also very quiet, it's true silence
                    context_ratio = context_quiet_count / (end_check - start_check)
                    if context_ratio > 0.7:  # 70% of context is quiet
                        silence_states.append(2)  # True silence
                    else:
                        silence_states.append(1)  # Pause
                else:
                    # Moderately quiet - check if it's a brief pause
                    # Look ahead to see if speech resumes soon
                    look_ahead = min(pause_buffer_frames, total_video_frames - frame_index - 1)
                    speech_resumes = False
                    
                    for look_idx in range(frame_index + 1, frame_index + 1 + look_ahead):
                        if look_idx < len(raw_energy_values):
                            look_rms, look_peak = raw_energy_values[look_idx]
                            if look_rms > pause_rms_threshold or look_peak > pause_peak_threshold:
                                speech_resumes = True
                                break
                    
                    if speech_resumes:
                        silence_states.append(1)  # Pause
                    else:
                        silence_states.append(2)  # Likely true silence

            # Create different templates and chunk sets
            silent_chunks = set()      # True silence chunks
            pause_chunks = set()       # Short pause chunks  
            silent_template = None     # Template for true silence
            pause_template = None      # Template for pauses (reduced animation)
            
            # Categorize chunks and create templates
            for frame_index in range(total_video_frames):
                if silence_states[frame_index] == 2:  # True silence
                    silent_chunks.add(frame_index)
                    if silent_template is None:
                        # Create a more neutral template for true silence
                        silent_template = torch.randn_like(audio_prompts[frame_index]) * 0.002
                elif silence_states[frame_index] == 1:  # Pause
                    pause_chunks.add(frame_index)
                    if pause_template is None:
                        # Use actual audio but reduced amplitude for pauses
                        pause_template = audio_prompts[frame_index].clone() * 0.3

            # Fallback templates
            if silent_template is None and len(audio_prompts) > 0:
                silent_template = torch.randn_like(audio_prompts[0]) * 0.002
            if pause_template is None and len(audio_prompts) > 0:
                pause_template = audio_prompts[0].clone() * 0.3

            audio_prompts = torch.cat(audio_prompts, dim=0)

            print(f"Processed {len(audio_prompts)} chunks for {total_video_frames} frames @ {video_fps}fps")
            print(f"Smart silence detection: {len(silent_chunks)} true silence, {len(pause_chunks)} pauses, {total_video_frames - len(silent_chunks) - len(pause_chunks)} speaking")
            
            # Return extended info including pause handling
            return audio_prompts, silent_chunks, silent_template, pause_chunks, pause_template

        except Exception as e:
            print(f"Error in whisper chunk processing with silence detection: {e}")
            import traceback
            traceback.print_exc()
            return [], set(), None, set(), None

    def _process_and_cache_audio(self, audio_path: str, fps: float, audio_source: int = 1):
        """Process audio upfront and cache chunks like realtime_inference.py"""
        try:
            # Convert MP3/other formats to WAV for precise timing
            wav_audio_path = ensure_wav_audio(audio_path)
            
            # Choose the right cache variables based on audio source
            if audio_source == 1:
                current_path = self._current_audio_path
                cached_chunks = self._whisper_chunks
            else:
                current_path = self._current_audio_path_2
                cached_chunks = self._whisper_chunks_2

            # Skip if already processed (check both original and WAV paths)
            if (current_path == audio_path or current_path == wav_audio_path) and cached_chunks is not None:
                return cached_chunks

            # Process whisper chunks with integrated silence detection using WAV
            whisper_chunks, silent_chunks, silent_template, pause_chunks, pause_template = self._get_whisper_chunks_with_silence_detection(wav_audio_path,
                                                                                                             fps,
                                                                                                             self.weight_dtype,
                                                                                                             self.device)

            # Cache the results in the appropriate variables
            if audio_source == 1:
                self._whisper_chunks = whisper_chunks
                self._current_audio_path = wav_audio_path  # Store WAV path
                self._silent_chunks = silent_chunks
                self._silent_template = silent_template
                self._pause_chunks = pause_chunks
                self._pause_template = pause_template
            else:
                self._whisper_chunks_2 = whisper_chunks
                self._current_audio_path_2 = wav_audio_path  # Store WAV path
                self._silent_chunks_2 = silent_chunks
                self._silent_template_2 = silent_template
                self._pause_chunks_2 = pause_chunks
                self._pause_template_2 = pause_template

            silent_status = f"{len(silent_chunks)} silent" if silent_chunks else "no silent"
            pause_status = f"{len(pause_chunks)} pause" if pause_chunks else "no pause"
            print(
                f"Cached {len(whisper_chunks)} audio chunks for source {audio_source}: {wav_audio_path} ({silent_status}, {pause_status})")
            return whisper_chunks

        except Exception as e:
            print(f"Audio processing failed for source {audio_source}: {e}")
            traceback.print_exc()
            return []

    def get_audio_chunk_for_frame(self, frame_index: int, audio_source: int = 1) -> tuple:
        """Get the appropriate audio chunk for a specific frame index with smart silence/pause handling"""
        try:
            # Handle negative frame indices (use 0 instead)
            if frame_index < 0:
                frame_index = 0

            # Choose the right audio chunks and templates
            chunks = self._whisper_chunks if audio_source == 1 else self._whisper_chunks_2
            silent_chunks = self._silent_chunks if audio_source == 1 else self._silent_chunks_2
            silent_template = self._silent_template if audio_source == 1 else self._silent_template_2
            pause_chunks = self._pause_chunks if audio_source == 1 else self._pause_chunks_2
            pause_template = self._pause_template if audio_source == 1 else self._pause_template_2

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

            # Check chunk type and return appropriate template/chunk
            is_true_silence = silent_chunks is not None and chunk_index in silent_chunks
            is_pause = pause_chunks is not None and chunk_index in pause_chunks
            
            if is_true_silence and silent_template is not None:
                # True silence - use silent template (neutral mouth position)
                return silent_template, True  # Return True to indicate silence
            elif is_pause and pause_template is not None:
                # Short pause - use pause template (reduced animation, maintains some mouth movement)
                return pause_template, False  # Return False as it's not true silence
            else:
                # Normal speech - use actual audio chunk
                audio_chunk = chunks[chunk_index]
                # Ensure we return a proper tensor
                if isinstance(audio_chunk, torch.Tensor):
                    return audio_chunk, False
                else:
                    return None, False

        except Exception as e:
            print(f"Error getting audio chunk for frame {frame_index}: {e}")
            import traceback
            traceback.print_exc()
            return None, False

    def pre_check(self) -> bool:
        """Download MuseTalk models if needed"""
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

        # Create proper mask using the same system as face_swapper
        self.masker = FaceMasker()

        # Ensure we have good defaults for lip syncing if settings are not configured
        face_mask_blur = state_manager.get_item('face_mask_blur') or 0.3
        self.face_mask_padding = state_manager.get_item('face_mask_padding') or (0, 0, 0, 0)
        self.face_mask_types = ['region' 'occlusion']
        self.face_mask_regions = state_manager.get_item('face_mask_regions') or ['mouth']

        # For lip syncing, we want more aggressive blur around mouth area
        # Increase blur amount for better feathering
        self.enhanced_blur = max(face_mask_blur, 0.4)  # Ensure minimum blur for smooth edges

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
        # Create proper mask using the same system as face_swapper
        if not self.masker:
            self.masker = FaceMasker()

        # Ensure we have good defaults for lip syncing if settings are not configured
        face_mask_blur = state_manager.get_item('face_mask_blur') or 0.3
        self.face_mask_padding = state_manager.get_item('face_mask_padding') or (0, 0, 0, 0)
        self.face_mask_types = ['region']
        self.face_mask_regions = state_manager.get_item('face_mask_regions') or ['mouth']

        # For lip syncing, we want more aggressive blur around mouth area
        # Increase blur amount for better feathering
        self.enhanced_blur = max(face_mask_blur, 0.4)  # Ensure minimum blur for smooth edges

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

    def sync_lip(self, target_face: Face, audio_chunk: torch.Tensor, temp_vision_frame: VisionFrame) -> VisionFrame:
        """Main lip sync method using MuseTalk with properly processed audio chunk"""
        try:
            if audio_chunk is None:
                audio_chunk = torch.randn(1, 50, 384, device=self.device, dtype=self.weight_dtype) * 0.05

            crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame,
                                                                            target_face.landmark_set.get('5/68'),
                                                                            'ffhq_512', (512, 512))

            # Calculate MuseTalk-style bounding box on the aligned frame
            face_landmark_68_transformed = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2),
                                                         affine_matrix).reshape(-1, 2)

            # Use MuseTalk methodology on transformed landmarks
            half_face_coord = face_landmark_68_transformed[29]  # nose tip area
            half_face_dist = numpy.max(face_landmark_68_transformed[:, 1]) - half_face_coord[1]
            min_upper_bond = 0
            upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)

            # Create bounding box from transformed landmarks
            x1 = int(numpy.min(face_landmark_68_transformed[:, 0]))
            y1 = int(upper_bond)
            x2 = int(numpy.max(face_landmark_68_transformed[:, 0]))
            y2 = int(numpy.max(face_landmark_68_transformed[:, 1]))

            # Add extra margin for v1.5 like original MuseTalk
            extra_margin = 10
            y2 = min(y2 + extra_margin, crop_vision_frame.shape[0])

            # Ensure valid bounding box
            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                # Use full crop as fallback
                bounding_box = create_bounding_box(face_landmark_68_transformed)
                x1, y1, x2, y2 = bounding_box
                y2 = min(y2 + extra_margin, crop_vision_frame.shape[0])

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face_crop = crop_vision_frame[y1:y2, x1:x2]

            # Resize using LANCZOS4 like original MuseTalk (with GPU acceleration if available)
            face_crop_resized = resize_gpu_or_cpu(face_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

            # Process audio chunk through PE like realtime_inference.py
            # Add batch dimension if missing (PE expects [batch, seq_len, d_model])
            if audio_chunk.dim() == 2:
                audio_chunk = audio_chunk.unsqueeze(0)  # Add batch dimension: [50, 384] -> [1, 50, 384]

            audio_features = self._musetalk_pe(audio_chunk.to(self.device))

            # MuseTalk inference - single step latent space inpainting
            latents = self._musetalk_vae.get_latents_for_unet(face_crop_resized)
            latents = latents.to(dtype=self.weight_dtype, device=self.device)
            audio_features = audio_features.to(dtype=self.weight_dtype, device=self.device)

            timesteps = torch.tensor([0], device=self.device, dtype=self.weight_dtype)

            # Single-step UNet inference (NOT diffusion - this is key!)
            with torch.no_grad():
                pred_latents = self._musetalk_unet.model(latents, timesteps,
                                                         encoder_hidden_states=audio_features).sample

            # Decode latents back to image
            result_frame = self._musetalk_vae.decode_latents(pred_latents)
            result_frame = result_frame[0]

            # Resize back to original face size and blend into aligned frame (with GPU acceleration if available)
            result_frame_resized = resize_gpu_or_cpu(result_frame.astype(numpy.uint8), (x2 - x1, y2 - y1))
            blended_frame = crop_vision_frame.copy()
            blended_frame[y1:y2, x1:x2] = result_frame_resized

            crop_mask = self.masker.create_combined_mask(
                self.face_mask_types,
                blended_frame.shape[:2][::-1],
                self.enhanced_blur,
                self.face_mask_padding,
                self.face_mask_regions,
                blended_frame,
                temp_vision_frame,
                target_face.landmark_set.get('5/68'),
                target_face
            )

            # # Apply additional gaussian blur to soften edges even more
            # if crop_mask.any():
            #     blur_kernel_size = max(3, int(blended_frame.shape[0] * 0.02))  # 2% of frame height
            #     if blur_kernel_size % 2 == 0:  # Ensure odd kernel size
            #         blur_kernel_size += 1
            #     crop_mask = cv2.GaussianBlur(crop_mask, (blur_kernel_size, blur_kernel_size), 0)

            crop_mask = crop_mask.clip(0, 1)

            # Paste back to original frame with proper masking
            paste_vision_frame = paste_back(temp_vision_frame, blended_frame, crop_mask, affine_matrix)

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
            
        # Clear cached audio data
        self._whisper_chunks = None
        self._whisper_chunks_2 = None
        self._current_audio_path = None
        self._current_audio_path_2 = None
        self._silent_chunks = None
        self._silent_chunks_2 = None
        self._silent_template = None
        self._silent_template_2 = None
        self._pause_chunks = None
        self._pause_chunks_2 = None
        self._pause_template = None
        self._pause_template_2 = None

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
                        target_vision_frame = self.sync_lip(target_face, audio_chunk, target_vision_frame)
                    except Exception as e:
                        logger.error(f"Error while syncing lip for face {i + 1} in 'many' mode: {e}", __name__)

        elif face_selector_mode == 'one':
            target_face = get_one_face(many_faces)
            if target_face:
                try:
                    audio_chunk, is_silent = self.get_audio_chunk_for_frame(frame_index, 1)
                    target_vision_frame = self.sync_lip(target_face, audio_chunk, target_vision_frame)
                except Exception as e:
                    logger.error(f"Error while syncing lip for 'one' mode: {e}", __name__)

        elif face_selector_mode == 'reference':
            if not reference_faces or len(reference_faces) == 0:
                logger.warn("No reference faces available for 'reference' mode", __name__)
                return target_vision_frame

            # Process each source's reference faces
            # reference_faces is Dict[int, List[Face]] where key = source_index, value = list of faces
            for source_index, ref_faces in reference_faces.items():
                if reference_faces is None or len(reference_faces) == 0:
                    continue

                # Map source index to audio source (source_index 0 → audio_source 1, etc.)
                if isinstance(source_index, str):
                    try:
                        source_index = int(source_index)
                    except ValueError:
                        logger.error(f"Invalid source index {source_index}, skipping", __name__)
                        continue
                audio_source = source_index + 1
                audio_chunk, is_silent = self.get_audio_chunk_for_frame(frame_index, audio_source)

                # Find similar faces
                try:
                    similar_faces = find_similar_faces(many_faces, ref_faces,
                                                       state_manager.get_item('reference_face_distance'))

                    if similar_faces:
                        for sim_idx, similar_face in enumerate(similar_faces):
                            try:
                                target_vision_frame = self.sync_lip(similar_face, audio_chunk, target_vision_frame)
                            except Exception as e:
                                logger.error(
                                    f"Error while syncing lip for similar face {sim_idx + 1} of reference source {source_index + 1}: {e}",
                                    __name__)
                except Exception as e:
                    logger.error(f"Error finding similar faces for reference source {source_index + 1}: {e}", __name__)

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
                reference_faces = get_reference_faces() if 'reference' in state_manager.get_item(
                    'face_selector_mode') else None

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
