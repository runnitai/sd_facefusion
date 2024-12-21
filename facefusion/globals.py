import os
from typing import List, Optional

from facefusion.choices import face_mask_regions
from facefusion.typing import LogLevel, VideoMemoryStrategy, FaceSelectorMode, FaceSelectorOrder, FaceAnalyserAge, \
    FaceAnalyserGender, FaceMaskType, FaceMaskRegion, OutputVideoEncoder, OutputVideoPreset, FaceDetectorModel, \
    FaceRecognizerModel, TempFrameFormat, Padding
from modules.paths_internal import default_output_dir

age_modifier_model: Optional[str] = "styleganex_age"
age_modifier_direction: Optional[str] = "0"
batch_size: Optional[int] = 4
# general
source_paths: Optional[List[str]] = None
target_path: Optional[str] = None
output_path: Optional[str] = os.path.join(default_output_dir, 'facefusion')
config_path: Optional[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'facefusion.ini'))
# misc
skip_download: Optional[bool] = False
headless: Optional[bool] = False
log_level: Optional[LogLevel] = ['info']
# execution
execution_providers: List[str] = ['tensorrt', 'cuda']
execution_thread_count: Optional[int] = 22
execution_queue_count: Optional[int] = 2
expression_restorer_model: Optional[str] = 'live_portrait'
face_editor_model: Optional[str] = 'live_portrait'
face_enhancer_model: Optional[str] = 'gfpgan_1.4'
frame_enhancer_blend: Optional[float] = 1.0
face_landmarker_model: Optional[str] = 'many'
face_swapper_model: Optional[str] = 'inswapper_128_fp16'
frame_colorizer_model: Optional[str] = 'ddcolor'
frame_enhancer_model: Optional[str] = 'real_esrgan_x2_fp16'
lip_syncer_model: Optional[str] = 'wav2lip_gan_96'
style_changer_model: Optional[str] = '3d'
face_swapper_pixel_boost: Optional[str] = "512x512"
# memory
video_memory_strategy: Optional[VideoMemoryStrategy] = "tolerant"
system_memory_limit: Optional[int] = 0
# face analyser
face_selector_order: Optional[FaceSelectorOrder] = 'best-worst'
face_selector_age_start: Optional[FaceAnalyserAge] = None
face_selector_age_end: Optional[FaceAnalyserAge] = None
face_selector_gender: Optional[FaceAnalyserGender] = None
face_selector_mode: Optional[FaceSelectorMode] = 'reference'
face_selector_race: Optional[str] = None

face_detector_model: Optional[FaceDetectorModel] = 'many'
face_detector_size: Optional[str] = "640x640"
face_detector_score: Optional[float] = 0.35
face_landmarker_score: Optional[float] = 0.35
face_detector_angles: Optional[List[int]] = [0]
face_recognizer_model: Optional[FaceRecognizerModel] = 'arcface_inswapper'
# face selector

frame_colorizer_blend: Optional[float] = 0.5

jobs_path: Optional[str] = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", 'jobs'))
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
output_audio_encoder: Optional[str] = 'aac'
output_image_quality: Optional[int] = 60
output_image_resolution: Optional[str] = None
output_video_encoder: Optional[OutputVideoEncoder] = 'libx264'
output_video_preset: Optional[OutputVideoPreset] = 'veryfast'
output_video_quality: Optional[int] = 60
output_video_resolution: Optional[str] = None
output_video_fps: Optional[float] = None
skip_audio: Optional[bool] = False
style_changer_target: Optional[str] = 'target'
style_changer_skip_head: Optional[bool] = False
style_transfer_model: Optional[str] = 'style_transfer'
style_transfer_source: Optional[str] = None
# frame processors
processors: List[str] = ["Face Swapper"]
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
ui_workflow: Optional[str] = "instant_runner"

