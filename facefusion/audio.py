from functools import lru_cache
from typing import Any, List, Optional

import numpy
import scipy
import torch
from numpy._typing import NDArray

from facefusion.ffmpeg import read_audio_buffer
from facefusion.filesystem import is_audio, resolve_relative_path
from facefusion.typing import Audio, AudioFrame, Fps, Mel, MelFilterBank, Spectrogram
from facefusion.voice_extractor import batch_extract_voice


@lru_cache(maxsize=128)
def read_static_audio(audio_path: str, fps: Fps) -> Optional[List[AudioFrame]]:
    return read_audio(audio_path, fps)


def read_audio(audio_path: str, fps: Fps) -> Optional[List[AudioFrame]]:
    sample_rate = 48000
    channel_total = 2

    if is_audio(audio_path):
        audio_buffer = read_audio_buffer(audio_path, sample_rate, channel_total)
        audio = numpy.frombuffer(audio_buffer, dtype=numpy.int16).reshape(-1, 2)
        audio = prepare_audio(audio)
        spectrogram = create_spectrogram(audio)
        audio_frames = extract_audio_frames(spectrogram, fps)
        return audio_frames
    return None


@lru_cache(maxsize=128)
def read_static_voice(audio_path: str, fps: Fps) -> Optional[List[AudioFrame]]:
    return read_voice(audio_path, fps)


def read_voice(audio_path: str, fps: Fps) -> Optional[List[AudioFrame]]:
    sample_rate = 48000
    channel_total = 2
    chunk_size = 240 * 1024
    step_size = 180 * 1024

    if is_audio(audio_path):
        audio_buffer = read_audio_buffer(audio_path, sample_rate, channel_total)
        audio = numpy.frombuffer(audio_buffer, dtype=numpy.int16).reshape(-1, 2)
        audio = batch_extract_voice(audio, chunk_size, step_size)
        audio = prepare_voice(audio)
        spectrogram = create_spectrogram(audio)
        audio_frames = extract_audio_frames(spectrogram, fps)
        return audio_frames
    return None


# NEW: Raw audio functions for MuseTalk
@lru_cache(maxsize=128)
def read_static_raw_audio(audio_path: str, fps: Fps) -> Optional[List[numpy.ndarray]]:
    """Read raw audio samples for MuseTalk processing"""
    return read_raw_audio(audio_path, fps)


def read_raw_audio(audio_path: str, fps: Fps) -> Optional[List[numpy.ndarray]]:
    """Read and process raw audio for MuseTalk at 16kHz"""
    if not is_audio(audio_path):
        return None
        
    # Read at original sample rate
    sample_rate = 48000
    channel_total = 2
    
    audio_buffer = read_audio_buffer(audio_path, sample_rate, channel_total)
    audio = numpy.frombuffer(audio_buffer, dtype=numpy.int16).reshape(-1, 2)
    
    # Convert to mono and normalize
    if audio.ndim > 1:
        audio = numpy.mean(audio, axis=1)
    audio = audio.astype(numpy.float32) / 32768.0  # Convert to float32 range [-1, 1]
    
    # Resample to 16kHz for MuseTalk/Whisper
    target_sample_rate = 16000
    audio = scipy.signal.resample(audio, int(len(audio) * target_sample_rate / sample_rate))
    
    # Apply pre-emphasis filter (standard for speech processing)
    audio = scipy.signal.lfilter([1.0, -0.97], [1.0], audio)
    
    # Extract frames based on FPS
    samples_per_frame = target_sample_rate // int(fps)  # e.g., 640 samples for 25fps
    audio_frames = []
    
    for i in range(0, len(audio) - samples_per_frame + 1, samples_per_frame):
        frame = audio[i:i + samples_per_frame]
        audio_frames.append(frame.astype(numpy.float32))
    
    return audio_frames


@lru_cache(maxsize=128) 
def read_static_raw_voice(audio_path: str, fps: Fps) -> Optional[List[numpy.ndarray]]:
    """Read raw voice samples for MuseTalk processing with voice extraction"""
    return read_raw_voice(audio_path, fps)


def read_raw_voice(audio_path: str, fps: Fps) -> Optional[List[numpy.ndarray]]:
    """Read and process raw voice audio for MuseTalk at 16kHz - simplified approach"""
    if not is_audio(audio_path):
        print(f"DEBUG: {audio_path} is not a valid audio file")
        return None
        
    # Simple approach - read directly and convert to 16kHz
    try:
        import soundfile
        audio_buffer, source_sample_rate = soundfile.read(audio_path)
        print(f"DEBUG: Read audio directly: shape {audio_buffer.shape}, sample_rate {source_sample_rate}, max: {numpy.max(numpy.abs(audio_buffer))}")
    except Exception as e:
        print(f"DEBUG: ERROR reading audio with soundfile: {e}")
        return None
    
    # Convert to mono if stereo
    if audio_buffer.ndim == 2:
        audio_buffer = numpy.mean(audio_buffer, axis=1)
        print(f"DEBUG: Converted to mono: shape {audio_buffer.shape}")
    
    # Convert to float32 and normalize to [-1, 1]
    audio_buffer = audio_buffer.astype(numpy.float32)
    max_val = numpy.max(numpy.abs(audio_buffer))
    if max_val > 1.0:
        audio_buffer = audio_buffer / max_val
        print(f"DEBUG: Normalized from max {max_val:.6f} to [-1, 1]")
    
    # Resample to 16kHz for Whisper
    target_sample_rate = 16000
    if source_sample_rate != target_sample_rate:
        import librosa
        audio_buffer = librosa.resample(audio_buffer, orig_sr=source_sample_rate, target_sr=target_sample_rate)
        print(f"DEBUG: Resampled from {source_sample_rate}Hz to {target_sample_rate}Hz: shape {audio_buffer.shape}")
    
    print(f"DEBUG: Final audio after basic processing: shape {audio_buffer.shape}, max: {numpy.max(numpy.abs(audio_buffer)):.6f}")
    
    # Extract frames based on FPS - simple approach
    samples_per_frame = target_sample_rate // int(fps)  # e.g., 640 samples for 25fps
    audio_frames = []
    
    print(f"DEBUG: Extracting frames with {samples_per_frame} samples per frame (fps={fps})")
    
    for i in range(0, len(audio_buffer) - samples_per_frame + 1, samples_per_frame):
        frame = audio_buffer[i:i + samples_per_frame]
        audio_frames.append(frame.astype(numpy.float32))
    
    print(f"DEBUG: Created {len(audio_frames)} audio frames")
    if audio_frames:
        print(f"DEBUG: First frame max: {numpy.max(numpy.abs(audio_frames[0])):.6f}")
        print(f"DEBUG: Last frame max: {numpy.max(numpy.abs(audio_frames[-1])):.6f}")
    
    return audio_frames


def get_raw_audio_frame(audio_path: str, fps: Fps, frame_number: int = 0) -> Optional[numpy.ndarray]:
    """Get raw audio frame for MuseTalk processing"""
    if is_audio(audio_path):
        audio_frames = read_static_raw_audio(audio_path, fps)
        if audio_frames and frame_number < len(audio_frames):
            return audio_frames[frame_number]
    return None


def get_raw_voice_frame(audio_path: str, fps: Fps, frame_number: int = 0) -> Optional[numpy.ndarray]:
    """Get raw voice frame for MuseTalk processing"""
    if is_audio(audio_path):
        voice_frames = read_static_raw_voice(audio_path, fps)
        if voice_frames and frame_number < len(voice_frames):
            frame = voice_frames[frame_number]
            print(f"DEBUG: get_raw_voice_frame - frame {frame_number}: shape {frame.shape if frame is not None else 'None'}, max: {numpy.max(numpy.abs(frame)) if frame is not None else 'None'}")
            return frame
        else:
            print(f"DEBUG: get_raw_voice_frame - No frames available or frame {frame_number} out of range. Total: {len(voice_frames) if voice_frames else 0}")
    else:
        print(f"DEBUG: get_raw_voice_frame - {audio_path} is not an audio file")
    return None


def create_empty_raw_audio_frame(fps: Fps = 25) -> numpy.ndarray:
    """Create empty raw audio frame for MuseTalk"""
    samples_per_frame = 16000 // int(fps)  # 16kHz / fps, ensure integer division
    return numpy.zeros(samples_per_frame, dtype=numpy.float32)


def get_audio_frame(audio_paths: List[str], fps: Fps, frame_number: int = 0) -> Optional[AudioFrame]:
    if not audio_paths:
        return None
    for audio_path in audio_paths:
        if is_audio(audio_path):
            audio_frames = read_static_audio(audio_path, fps)
            if audio_frames and frame_number in range(len(audio_frames)):
                return audio_frames[frame_number]
    return None


def get_voice_frame(audio_paths: str, fps: Fps, frame_number: int = 0) -> Optional[AudioFrame]:
    if not audio_paths:
        return None
    for audio_path in audio_paths:
        if is_audio(audio_path):
            voice_frames = read_static_voice(audio_path, fps)
            if frame_number in range(len(voice_frames)):
                return voice_frames[frame_number]
    return None


def create_empty_audio_frame() -> AudioFrame:
    mel_filter_total = 80
    step_size = 16
    audio_frame = numpy.zeros((mel_filter_total, step_size)).astype(numpy.int16)
    return audio_frame


def prepare_audio(audio: Audio) -> Audio:
    if audio.ndim > 1:
        audio = numpy.mean(audio, axis=1)
    audio = audio / numpy.max(numpy.abs(audio), axis=0)
    audio = scipy.signal.lfilter([1.0, -0.97], [1.0], audio)
    return audio


def prepare_voice(audio: Audio) -> Audio:
    sample_rate = 48000
    resample_rate = 16000

    audio = scipy.signal.resample(audio, int(len(audio) * resample_rate / sample_rate))
    audio = prepare_audio(audio)
    return audio


def convert_hertz_to_mel(hertz: float) -> float:
    return 2595 * numpy.log10(1 + hertz / 700)


def convert_mel_to_hertz(mel: Mel) -> NDArray[Any]:
    return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filter_bank() -> MelFilterBank:
    mel_filter_total = 80
    mel_bin_total = 800
    sample_rate = 16000
    min_frequency = 55.0
    max_frequency = 7600.0
    mel_filter_bank = numpy.zeros((mel_filter_total, mel_bin_total // 2 + 1))
    mel_frequency_range = numpy.linspace(convert_hertz_to_mel(min_frequency), convert_hertz_to_mel(max_frequency),
                                         mel_filter_total + 2)
    indices = numpy.floor((mel_bin_total + 1) * convert_mel_to_hertz(mel_frequency_range) / sample_rate).astype(
        numpy.int16)

    for index in range(mel_filter_total):
        start = indices[index]
        end = indices[index + 1]
        mel_filter_bank[index, start:end] = scipy.signal.windows.triang(end - start)
    return mel_filter_bank


def create_spectrogram(audio: Audio) -> Spectrogram:
    mel_bin_total = 800
    mel_bin_overlap = 600
    mel_filter_bank = create_mel_filter_bank()
    spectrogram = scipy.signal.stft(audio, nperseg=mel_bin_total, nfft=mel_bin_total, noverlap=mel_bin_overlap)[2]
    spectrogram = numpy.dot(mel_filter_bank, numpy.abs(spectrogram))
    return spectrogram


def extract_audio_frames(spectrogram: Spectrogram, fps: Fps) -> List[AudioFrame]:
    mel_filter_total = 80
    step_size = 16
    audio_frames = []
    indices = numpy.arange(0, spectrogram.shape[1], mel_filter_total / fps).astype(numpy.int16)
    indices = indices[indices >= step_size]

    for index in indices:
        start = max(0, index - step_size)
        audio_frames.append(spectrogram[:, start:index])
    return audio_frames
