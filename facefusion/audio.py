from functools import lru_cache
from typing import Any, List, Optional

import numpy
import scipy
from numpy._typing import NDArray

from facefusion import logger
from facefusion.ffmpeg import read_audio_buffer
from facefusion.filesystem import is_audio
from facefusion.typing import Audio, AudioFrame, Fps, Mel, MelFilterBank, Spectrogram
from facefusion.workers.classes.voice_extractor import VoiceExtractor


@lru_cache(maxsize=128)
def read_static_audio(audio_path: str, fps: Fps) -> Optional[List[AudioFrame]]:
    return read_audio(audio_path, fps)


def read_audio(audio_path: str, fps: Fps) -> Optional[List[AudioFrame]]:
    sample_rate = 48000
    channel_total = 2

    if not is_audio(audio_path):
        logger.error(f"File is not recognized as audio: {audio_path}", __name__)
        return None

    try:
        logger.info(f"Reading audio file: {audio_path}", __name__)
        audio_buffer = read_audio_buffer(audio_path, sample_rate, channel_total)
        
        if audio_buffer is None or len(audio_buffer) == 0:
            logger.error(f"Failed to read audio buffer from {audio_path}", __name__)
            # Try to use direct WAV reading for WAV files
            if audio_path.lower().endswith('.wav'):
                return read_wav_directly(audio_path, fps)
            return None
            
        try:
            # If we get an empty buffer, try direct WAV reading
            if len(audio_buffer) < 1000:  # Very small buffer, likely empty
                logger.warn(f"Audio buffer is too small ({len(audio_buffer)} bytes), trying direct WAV reading", __name__)
                if audio_path.lower().endswith('.wav'):
                    return read_wav_directly(audio_path, fps)
                    
            audio = numpy.frombuffer(audio_buffer, dtype=numpy.int16).reshape(-1, 2)
            
            if audio.size == 0:
                logger.error(f"Audio buffer converted to empty array: {audio_path}", __name__)
                if audio_path.lower().endswith('.wav'):
                    return read_wav_directly(audio_path, fps)
                return None
                
            audio = prepare_audio(audio)
            logger.info(f"Audio prepared, shape: {audio.shape}", __name__)
            
            spectrogram = create_spectrogram(audio)
            logger.info(f"Spectrogram created, shape: {spectrogram.shape}", __name__)
            
            audio_frames = extract_audio_frames(spectrogram, fps)
            logger.info(f"Audio frames extracted: {len(audio_frames)}", __name__)
            
            return audio_frames
        except Exception as e:
            logger.error(f"Error processing audio data: {str(e)}", __name__)
            # Fall back to direct WAV reading for WAV files
            if audio_path.lower().endswith('.wav'):
                return read_wav_directly(audio_path, fps)
            return None
    except Exception as e:
        logger.error(f"Error reading audio file {audio_path}: {str(e)}", __name__)
        return None


def read_wav_directly(audio_path: str, fps: Fps) -> Optional[List[AudioFrame]]:
    """Alternative method to read WAV files directly using scipy"""
    try:
        logger.info(f"Trying direct WAV reading: {audio_path}", __name__)
        from scipy.io import wavfile
        
        sample_rate, audio_data = wavfile.read(audio_path)
        logger.info(f"Read WAV file with sample rate {sample_rate}, shape: {audio_data.shape}", __name__)
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = numpy.mean(audio_data, axis=1)
            
        # Normalize and prepare
        audio_data = audio_data.astype(numpy.float32)
        
        # Check if audio is too quiet and amplify if needed
        audio_max = numpy.max(numpy.abs(audio_data))
        logger.info(f"Audio max absolute value: {audio_max}", __name__)
        
        if 0 < audio_max < 0.01:
            # Very quiet audio, amplify more aggressively
            amplification = min(0.5 / audio_max, 100.0)
            logger.info(f"Amplifying quiet audio by {amplification}x", __name__)
            audio_data = audio_data * amplification
        elif audio_max > 0:
            # Normal normalization
            audio_data = audio_data / audio_max
            
        # Apply filter (high-pass to remove DC component)
        audio_data = scipy.signal.lfilter([1.0, -0.97], [1.0], audio_data)
        
        # Create spectrogram and extract frames
        spectrogram = create_spectrogram(audio_data)
        logger.info(f"Created spectrogram with shape: {spectrogram.shape}", __name__)
        
        audio_frames = extract_audio_frames(spectrogram, fps)
        logger.info(f"Direct WAV reading successful, got {len(audio_frames)} frames", __name__)
        
        return audio_frames
    except Exception as e:
        logger.error(f"Direct WAV reading failed: {str(e)}", __name__)
        return None


@lru_cache(maxsize=128)
def read_static_voice(audio_path: str, fps: Fps) -> Optional[List[AudioFrame]]:
    return read_voice(audio_path, fps)


def read_voice(audio_path: str, fps: Fps) -> Optional[List[AudioFrame]]:
    extractor = VoiceExtractor()
    sample_rate = 48000
    channel_total = 2
    chunk_size = 240 * 1024
    step_size = 180 * 1024

    if not is_audio(audio_path):
        logger.error(f"File is not recognized as audio: {audio_path}", __name__)
        return None
        
    try:
        logger.info(f"Reading voice from audio file: {audio_path}", __name__)
        audio_buffer = read_audio_buffer(audio_path, sample_rate, channel_total)
        
        if audio_buffer is None or len(audio_buffer) == 0:
            logger.error(f"Failed to read audio buffer for voice extraction: {audio_path}", __name__)
            # Try to use direct WAV reading for WAV files
            if audio_path.lower().endswith('.wav'):
                return read_wav_voice_directly(audio_path, fps)
            return None
        
        try:
            # If we get an empty buffer, try direct WAV reading
            if len(audio_buffer) < 1000:  # Very small buffer, likely empty
                logger.warn(f"Audio buffer for voice is too small ({len(audio_buffer)} bytes), trying direct WAV reading", __name__)
                if audio_path.lower().endswith('.wav'):
                    return read_wav_voice_directly(audio_path, fps)
            
            audio = numpy.frombuffer(audio_buffer, dtype=numpy.int16).reshape(-1, 2)
            
            if audio.size == 0:
                logger.error(f"Audio buffer converted to empty array for voice extraction: {audio_path}", __name__)
                if audio_path.lower().endswith('.wav'):
                    return read_wav_voice_directly(audio_path, fps)
                return None
                
            audio = extractor.batch_extract_voice(audio, chunk_size, step_size)
            logger.info(f"Voice extracted, shape: {audio.shape}", __name__)
            
            audio = prepare_voice(audio)
            logger.info(f"Voice prepared, shape: {audio.shape}", __name__)
            
            spectrogram = create_spectrogram(audio)
            logger.info(f"Voice spectrogram created, shape: {spectrogram.shape}", __name__)
            
            audio_frames = extract_audio_frames(spectrogram, fps)
            logger.info(f"Voice frames extracted: {len(audio_frames)}", __name__)
            
            return audio_frames
        except Exception as e:
            logger.error(f"Error processing voice data: {str(e)}", __name__)
            # Fall back to direct WAV reading for WAV files
            if audio_path.lower().endswith('.wav'):
                return read_wav_voice_directly(audio_path, fps)
            return None
    except Exception as e:
        logger.error(f"Error reading audio file for voice extraction {audio_path}: {str(e)}", __name__)
        return None


def read_wav_voice_directly(audio_path: str, fps: Fps) -> Optional[List[AudioFrame]]:
    """Alternative method to read WAV files directly for voice extraction"""
    try:
        logger.info(f"Trying direct WAV reading for voice: {audio_path}", __name__)
        from scipy.io import wavfile
        
        sample_rate, audio_data = wavfile.read(audio_path)
        logger.info(f"Read WAV file for voice with sample rate {sample_rate}, shape: {audio_data.shape}", __name__)
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = numpy.mean(audio_data, axis=1)
            
        # Normalize and prepare for voice
        audio_data = audio_data.astype(numpy.float32)
        
        # Check if audio is too quiet and amplify if needed
        audio_max = numpy.max(numpy.abs(audio_data))
        logger.info(f"Voice audio max absolute value: {audio_max}", __name__)
        
        if 0 < audio_max < 0.01:
            # Very quiet audio, amplify more aggressively
            amplification = min(0.5 / audio_max, 100.0)
            logger.info(f"Amplifying quiet voice audio by {amplification}x", __name__)
            audio_data = audio_data * amplification
        elif audio_max > 0:
            # Normal normalization
            audio_data = audio_data / audio_max
            
        # Apply filter (high-pass to remove DC component)
        audio_data = scipy.signal.lfilter([1.0, -0.97], [1.0], audio_data)
        
        # Resample for voice processing
        resample_rate = 16000
        audio_data = scipy.signal.resample(audio_data, int(len(audio_data) * resample_rate / sample_rate))
        
        # Create spectrogram and extract frames
        spectrogram = create_spectrogram(audio_data)
        logger.info(f"Created voice spectrogram with shape: {spectrogram.shape}", __name__)
        
        audio_frames = extract_audio_frames(spectrogram, fps)
        logger.info(f"Direct WAV voice reading successful, got {len(audio_frames)} frames with magnitudes: {[numpy.sum(numpy.abs(f)) for f in audio_frames[:3]]}", __name__)
        
        return audio_frames
    except Exception as e:
        logger.error(f"Direct WAV voice reading failed: {str(e)}", __name__)
        return None


def get_audio_frame(audio_path: str, fps: Fps, frame_number: int = 0) -> Optional[AudioFrame]:
    if is_audio(audio_path):
        audio_frames = read_static_audio(audio_path, fps)
        if audio_frames and frame_number in range(len(audio_frames)):
            return audio_frames[frame_number]
    return None


def get_voice_frame(audio_path: str, fps: Fps, frame_number: int = 0) -> Optional[AudioFrame]:
    if is_audio(audio_path):
        try:
            logger.info(f"Getting voice frame from {audio_path} for frame {frame_number}, fps {fps}", __name__)
            voice_frames = read_static_voice(audio_path, fps)
            
            if voice_frames is None:
                logger.error(f"No voice frames returned from read_static_voice for {audio_path}", __name__)
                return create_empty_audio_frame()
                
            if len(voice_frames) == 0:
                logger.error(f"Empty voice frames list for {audio_path}", __name__)
                return create_empty_audio_frame()
                
            if frame_number >= len(voice_frames):
                logger.warn(f"Frame number {frame_number} exceeds available frames {len(voice_frames)}", __name__)
                # Use the last available frame instead of returning None
                frame_number = len(voice_frames) - 1
                
            result = voice_frames[frame_number]
            
            # Check if result is valid
            if result is None:
                logger.error(f"Voice frame at position {frame_number} is None", __name__)
                return create_empty_audio_frame()
                
            if not isinstance(result, numpy.ndarray):
                logger.error(f"Voice frame has invalid type: {type(result)}", __name__)
                return create_empty_audio_frame()
                
            if result.size == 0:
                logger.error(f"Voice frame is empty (size=0)", __name__)
                return create_empty_audio_frame()
                
            # Even if the audio is extremely quiet, use it anyway as long as it's valid
            # Just log a warning for very quiet frames
            audio_magnitude = numpy.sum(numpy.abs(result))
            if audio_magnitude < 1e-6:
                logger.warn(f"Voice frame has very low magnitude: {audio_magnitude}", __name__)
                # Amplify very quiet audio to ensure it's detectable
                if audio_magnitude > 0:
                    amplification = max(1e-6 / audio_magnitude, 2.0)
                    result = result * amplification
                    logger.info(f"Amplified quiet audio by {amplification}x to {numpy.sum(numpy.abs(result))}", __name__)
                
            return result
        except Exception as e:
            logger.error(f"Error getting voice frame for {audio_path}: {str(e)}", __name__)
    
    return create_empty_audio_frame()


def create_empty_audio_frame() -> AudioFrame:
    mel_filter_total = 80
    step_size = 16
    audio_frame = numpy.zeros((mel_filter_total, step_size)).astype(numpy.float32)
    return audio_frame


def prepare_audio(audio: Audio) -> Audio:
    if audio.ndim > 1:
        audio = numpy.mean(audio, axis=1)
    max_value = numpy.max(numpy.abs(audio), axis=0)
    if max_value > 0:
        audio = audio / max_value
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
    
    # Add safeguard for empty audio
    if len(audio) < mel_bin_total:
        logger.warn(f"Audio too short ({len(audio)} samples), padding to required length", __name__)
        audio = numpy.pad(audio, (0, mel_bin_total - len(audio)), 'constant')
        
    try:
        spectrogram = scipy.signal.stft(audio, nperseg=mel_bin_total, nfft=mel_bin_total, noverlap=mel_bin_overlap)[2]
        spectrogram = numpy.dot(mel_filter_bank, numpy.abs(spectrogram))
        return spectrogram
    except Exception as e:
        logger.error(f"Error creating spectrogram: {str(e)}", __name__)
        # Return an empty spectrogram as fallback
        return numpy.zeros((80, 16), dtype=numpy.float32)


def extract_audio_frames(spectrogram: Spectrogram, fps: Fps) -> List[AudioFrame]:
    try:
        step = fps / 25
        spec_len = spectrogram.shape[1]
        audios_frames = []
        
        if step == 0:
            step = 1  # Safeguard against division by zero
            
        logger.info(f"Extracting audio frames from spectrogram of shape {spectrogram.shape}, step {step}", __name__)
        
        if spec_len <= 16:
            # Special handling for very short audio - duplicate to fill the frame
            logger.warn(f"Spectrogram length {spec_len} is too short, will pad", __name__)
            audio_frame = numpy.zeros((80, 16)).astype(numpy.float32)
            audio_frame[:, :spec_len] = spectrogram
            # Duplicate the content to fill the entire frame
            if spec_len > 0:
                for i in range(spec_len, 16):
                    audio_frame[:, i] = audio_frame[:, i % spec_len]
            audios_frames.append(audio_frame)
            return audios_frames
            
        # Normal processing for longer spectrograms
        for i in range(0, 999999):
            start_idx = int(i * step)
            if start_idx + 16 > spec_len:
                break
                
            audio_frame = spectrogram[:, start_idx: start_idx + 16]
            
            # Ensure the frame has the correct shape
            if audio_frame.shape[1] < 16:
                logger.warn(f"Padding audio frame from shape {audio_frame.shape}", __name__)
                padded = numpy.zeros((80, 16)).astype(numpy.float32)
                padded[:, :audio_frame.shape[1]] = audio_frame
                audio_frame = padded
                
            # Check if the frame is too quiet and amplify if needed
            frame_magnitude = numpy.sum(numpy.abs(audio_frame))
            if 0 < frame_magnitude < 1e-6:
                logger.info(f"Amplifying quiet audio frame at position {i} with magnitude {frame_magnitude}", __name__)
                amplification = max(1e-6 / frame_magnitude, 2.0)
                audio_frame = audio_frame * amplification
                
            # Handle NaN values
            if numpy.isnan(audio_frame).any():
                logger.warn(f"NaN values found in audio frame at position {i}, replacing with zeros", __name__)
                audio_frame = numpy.nan_to_num(audio_frame)
                
            audios_frames.append(audio_frame)
        
        if len(audios_frames) == 0:
            logger.warn("No audio frames extracted, creating a default frame", __name__)
            audios_frames.append(create_empty_audio_frame())
            
        logger.info(f"Extracted {len(audios_frames)} audio frames", __name__)
        return audios_frames
    except Exception as e:
        logger.error(f"Error extracting audio frames: {str(e)}", __name__)
        return [create_empty_audio_frame()]
