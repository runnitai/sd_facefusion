import os
from typing import List, Optional

from facefusion.typing import LogLevel, FaceSelectorMode, FaceAnalyserOrder, FaceAnalyserAge, FaceAnalyserGender, \
    FaceMaskType, OutputVideoEncoder, FaceDetectorModel, FaceRecognizerModel, TempFrameFormat, Padding, FaceMaskRegion, \
    OutputVideoPreset, VideoMemoryStrategy
from facefusion.choices import face_mask_regions
from modules.paths_internal import script_path

# general
source_paths: Optional[List[str]] = None
target_path: Optional[str] = None
restricted_path: Optional[str] = None
output_path: Optional[str] = os.path.join(script_path, "outputs", "facefusion")
# misc
skip_download: Optional[bool] = False
headless: Optional[bool] = False
log_level: Optional[LogLevel] = ['info']
# execution
execution_providers: List[str] = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'})]

execution_thread_count: Optional[int] = 32
execution_queue_count: Optional[int] = 2
video_memory_strategy: Optional[VideoMemoryStrategy] = "tolerant"
system_memory_limit: Optional[int] = None
# face analyser
face_analyser_order: Optional[FaceAnalyserOrder] = 'best-worst'
face_analyser_age: Optional[FaceAnalyserAge] = None
face_analyser_gender: Optional[FaceAnalyserGender] = None
face_detector_model: Optional[FaceDetectorModel] = 'retinaface'
face_detector_size: Optional[str] = "640x640"
face_detector_score: Optional[float] = 0.4
face_recognizer_model: Optional[FaceRecognizerModel] = 'arcface_inswapper'
# face selector
face_selector_mode: Optional[FaceSelectorMode] = 'reference'
reference_face_position: Optional[int] = 0
reference_face_distance: Optional[float] = 0.75
reference_frame_number: Optional[int] = 0
reference_face_dict: Optional[dict] = {}
# face mask
face_mask_types: Optional[List[FaceMaskType]] = ['box', 'region', 'occlusion']
mask_enabled_times: Optional[List[int]] = []
mask_disabled_times: Optional[List[int]] = [0]
face_mask_blur: Optional[float] = 0.3
face_mask_padding: Optional[Padding] = (0, 0, 0, 0)
face_mask_regions: Optional[List[FaceMaskRegion]] = face_mask_regions
# frame extraction
trim_frame_start: Optional[int] = None
trim_frame_end: Optional[int] = None
temp_frame_format: Optional[TempFrameFormat] = 'png'
temp_frame_quality: Optional[int] = 100
keep_temp: Optional[bool] = False
# output creation
output_image_quality: Optional[int] = 60
output_video_encoder: Optional[OutputVideoEncoder] = 'h264_nvenc'
output_video_preset: Optional[OutputVideoPreset] = 'veryfast'
output_video_quality: Optional[int] = 60
output_video_resolution: Optional[str] = None
output_video_fps: Optional[float] = None
skip_audio: Optional[bool] = False

# frame processors
frame_processors: List[str] = ["face_swapper"]
# uis
ui_layouts: List[str] = ["default"]
