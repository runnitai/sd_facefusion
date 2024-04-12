import os

from facefusion.typing import LogLevel, VideoMemoryStrategy, FaceSelectorMode, FaceAnalyserOrder, FaceAnalyserAge, \
    FaceAnalyserGender, FaceMaskType, FaceMaskRegion, OutputVideoEncoder, OutputVideoPreset, FaceDetectorModel, \
    FaceRecognizerModel, TempFrameFormat, Padding
from typing import List, Optional


from modules.paths_internal import script_path
from facefusion.choices import face_mask_regions

# general
source_paths: Optional[List[str]] = None
target_path: Optional[str] = None
output_path: Optional[str] = os.path.join(script_path, "outputs", "facefusion")
# misc
skip_download: Optional[bool] = False
headless: Optional[bool] = False
log_level: Optional[LogLevel] = ['info']
# execution
execution_providers: List[str] = ['CUDAExecutionProvider']
execution_thread_count: Optional[int] = 32
execution_queue_count: Optional[int] = 2
# memory
video_memory_strategy: Optional[VideoMemoryStrategy] = "tolerant"
system_memory_limit: Optional[int] = 0
# face analyser
face_analyser_order: Optional[FaceAnalyserOrder] = 'best-worst'
face_analyser_age: Optional[FaceAnalyserAge] = None
face_analyser_gender: Optional[FaceAnalyserGender] = None
face_detector_model: Optional[FaceDetectorModel] = 'many'
face_detector_size: Optional[str] = "640x640"
face_detector_score: Optional[float] = 0.35
face_landmarker_score: Optional[float] = 0.35
face_recognizer_model: Optional[FaceRecognizerModel] = 'arcface_inswapper'
# face selector
face_selector_mode: Optional[FaceSelectorMode] = 'reference'
reference_face_position: Optional[int] = 0
reference_face_distance: Optional[float] = 0.75
reference_frame_number: Optional[int] = 0
# face mask
face_mask_types: Optional[List[FaceMaskType]] = ['box', 'region', 'occlusion']
face_mask_blur: Optional[float] = 0.3
face_mask_padding: Optional[Padding] = (0, 0, 0, 0)
face_mask_regions: Optional[List[FaceMaskRegion]] = face_mask_regions
# frame extraction
trim_frame_start: Optional[int] = None
trim_frame_end: Optional[int] = None
temp_frame_format: Optional[TempFrameFormat] = 'png'
keep_temp: Optional[bool] = False
# output creation
output_image_quality: Optional[int] = 60
output_image_resolution: Optional[str] = None
output_video_encoder: Optional[OutputVideoEncoder] = 'libx264'
output_video_preset: Optional[OutputVideoPreset] = 'veryfast'
output_video_quality: Optional[int] = 60
output_video_resolution: Optional[str] = None
output_video_fps: Optional[float] = None
skip_audio: Optional[bool] = False
# frame processors
frame_processors: List[str] = ["face_swapper"]
# uis
ui_layouts: List[str] = ["default"]

# Custom elements for AUTO extension
mask_disabled_times: Optional[List[int]] = [0]
mask_enabled_times: Optional[List[int]] = []
reference_face_dict: Optional[dict] = {}
reference_face_dict_2: Optional[dict] = {}
restricted_path: Optional[str] = None
source_paths_2: Optional[List[str]] = None
sync_video_lip: Optional[bool] = False

