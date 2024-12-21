from typing import Tuple

import numpy
import scipy

from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import thread_semaphore
from facefusion.typing import Audio, AudioChunk, ModelSet
from facefusion.workers.base_worker import BaseWorker


class VoiceExtractor(BaseWorker):
    MODEL_SET: ModelSet = \
        {
            'kim_vocal_2':
                {
                    'hashes':
                        {
                            'voice_extractor':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/kim_vocal_2.hash',
                                    'path': resolve_relative_path('../.assets/models/kim_vocal_2.hash')
                                }
                        },
                    'sources':
                        {
                            'voice_extractor':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/kim_vocal_2.onnx',
                                    'path': resolve_relative_path('../.assets/models/kim_vocal_2.onnx')
                                }
                        }
                }
        }

    default_model = 'kim_vocal_2'
    multi_model = False
    model_key = None

    def forward(self, temp_audio_chunk: AudioChunk) -> AudioChunk:
        voice_extractor = self.get_inference_pool().get('voice_extractor')

        with thread_semaphore():
            temp_audio_chunk = voice_extractor.run(None,
                                                   {
                                                       'input': temp_audio_chunk
                                                   })[0]

        return temp_audio_chunk

    def batch_extract_voice(self, audio: Audio, chunk_size: int, step_size: int) -> Audio:
        temp_audio = numpy.zeros((audio.shape[0], 2)).astype(numpy.float32)
        temp_chunk = numpy.zeros((audio.shape[0], 2)).astype(numpy.float32)

        for start in range(0, audio.shape[0], step_size):
            end = min(start + chunk_size, audio.shape[0])
            temp_audio[start:end, ...] += self.extract_voice(audio[start:end, ...])
            temp_chunk[start:end, ...] += 1

        audio = temp_audio / temp_chunk
        return audio

    def extract_voice(self, temp_audio_chunk: AudioChunk) -> AudioChunk:
        voice_extractor = self.get_inference_pool().get('voice_extractor')
        chunk_size = (voice_extractor.get_inputs()[0].shape[3] - 1) * 1024
        trim_size = 3840
        temp_audio_chunk, pad_size = self.prepare_audio_chunk(temp_audio_chunk.T, chunk_size, trim_size)
        temp_audio_chunk = self.decompose_audio_chunk(temp_audio_chunk, trim_size)
        temp_audio_chunk = self.forward(temp_audio_chunk)
        temp_audio_chunk = self.compose_audio_chunk(temp_audio_chunk, trim_size)
        temp_audio_chunk = self.normalize_audio_chunk(temp_audio_chunk, chunk_size, trim_size, pad_size)
        return temp_audio_chunk

    def prepare_audio_chunk(self, temp_audio_chunk: AudioChunk, chunk_size: int, trim_size: int) -> Tuple[
        AudioChunk, int]:
        step_size = chunk_size - 2 * trim_size
        pad_size = step_size - temp_audio_chunk.shape[1] % step_size
        audio_chunk_size = temp_audio_chunk.shape[1] + pad_size
        temp_audio_chunk = temp_audio_chunk.astype(numpy.float32) / numpy.iinfo(numpy.int16).max
        temp_audio_chunk = numpy.pad(temp_audio_chunk, ((0, 0), (trim_size, trim_size + pad_size)))
        temp_audio_chunks = []

        for index in range(0, audio_chunk_size, step_size):
            temp_audio_chunks.append(temp_audio_chunk[:, index:index + chunk_size])

        temp_audio_chunk = numpy.concatenate(temp_audio_chunks, axis=0)
        temp_audio_chunk = temp_audio_chunk.reshape((-1, chunk_size))
        return temp_audio_chunk, pad_size

    def decompose_audio_chunk(self, temp_audio_chunk: AudioChunk, trim_size: int) -> AudioChunk:
        frame_size = 7680
        frame_overlap = 6656
        frame_total = 3072
        bin_total = 256
        channel_total = 4
        window = scipy.signal.windows.hann(frame_size)
        temp_audio_chunk = \
            scipy.signal.stft(temp_audio_chunk, nperseg=frame_size, noverlap=frame_overlap, window=window)[2]
        temp_audio_chunk = numpy.stack((numpy.real(temp_audio_chunk), numpy.imag(temp_audio_chunk)), axis=-1).transpose(
            (0, 3, 1, 2))
        temp_audio_chunk = temp_audio_chunk.reshape(-1, 2, 2, trim_size + 1, bin_total).reshape(-1, channel_total,
                                                                                                trim_size + 1,
                                                                                                bin_total)
        temp_audio_chunk = temp_audio_chunk[:, :, :frame_total]
        temp_audio_chunk /= numpy.sqrt(1.0 / window.sum() ** 2)
        return temp_audio_chunk

    def compose_audio_chunk(self, temp_audio_chunk: AudioChunk, trim_size: int) -> AudioChunk:
        frame_size = 7680
        frame_overlap = 6656
        frame_total = 3072
        bin_total = 256
        window = scipy.signal.windows.hann(frame_size)
        temp_audio_chunk = numpy.pad(temp_audio_chunk, ((0, 0), (0, 0), (0, trim_size + 1 - frame_total), (0, 0)))
        temp_audio_chunk = temp_audio_chunk.reshape(-1, 2, trim_size + 1, bin_total).transpose((0, 2, 3, 1))
        temp_audio_chunk = temp_audio_chunk[:, :, :, 0] + 1j * temp_audio_chunk[:, :, :, 1]
        temp_audio_chunk = \
            scipy.signal.istft(temp_audio_chunk, nperseg=frame_size, noverlap=frame_overlap, window=window)[1]
        temp_audio_chunk *= numpy.sqrt(1.0 / window.sum() ** 2)
        return temp_audio_chunk

    def normalize_audio_chunk(self, temp_audio_chunk: AudioChunk, chunk_size: int, trim_size: int,
                              pad_size: int) -> AudioChunk:
        temp_audio_chunk = temp_audio_chunk.reshape((-1, 2, chunk_size))
        temp_audio_chunk = temp_audio_chunk[:, :, trim_size:-trim_size].transpose(1, 0, 2)
        temp_audio_chunk = temp_audio_chunk.reshape(2, -1)[:, :-pad_size].T
        return temp_audio_chunk
