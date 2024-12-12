import json
import os
from typing import List, Optional

from facefusion.choices import face_mask_regions
from facefusion.memory import tune_performance
# Assuming the necessary imports are available in the environment:
from facefusion.typing import (
    FaceSelectorOrder, FaceAnalyserAge,
    FaceAnalyserGender, TempFrameFormat, OutputVideoEncoder, FaceSelectorMode, FaceDetectorModel, FaceRecognizerModel,
    Padding, FaceMaskType, FaceMaskRegion, LogLevel, OutputVideoPreset
)
from modules.paths_internal import script_path

execution_thread_count, execution_queue_count, video_memory_strategy = tune_performance()
        
class JobParams:
    def __init__(self):
        self.id = 0
        # general
        self.source_paths: Optional[List[str]] = None
        self.target_path: Optional[str] = None
        self.output_path: Optional[str] = os.path.join(script_path, "outputs", "facefusion")
        # misc
        self.skip_download: Optional[bool] = False
        self.headless: Optional[bool] = False
        self.log_level: Optional[LogLevel] = ['info']
        # execution
        self.execution_providers: List[str] = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'})]
        self.execution_thread_count: Optional[int] = execution_thread_count
        self.execution_queue_count: Optional[int] = execution_queue_count
        # memory
        self.video_memory_strategy: Optional[str] = video_memory_strategy
        self.max_memory: Optional[int] = None
        # face analyser
        self.face_analyser_order: Optional[FaceSelectorOrder] = 'best-worst'
        self.face_analyser_age: Optional[FaceAnalyserAge] = None
        self.face_analyser_gender: Optional[FaceAnalyserGender] = None
        self.face_detector_model: Optional[FaceDetectorModel] = 'yoloface'
        self.face_detector_size: Optional[str] = "640x640"
        self.face_detector_score: Optional[float] = 0.4
        self.face_landmarker_score: Optional[float] = 0.4
        self.face_recognizer_model: Optional[FaceRecognizerModel] = 'arcface_inswapper'
        # face selector
        self.face_selector_mode: Optional[FaceSelectorMode] = 'reference'
        self.reference_face_position: Optional[int] = 0
        self.reference_face_distance: Optional[float] = 0.75
        self.reference_frame_number: Optional[int] = 0
        # face mask
        self.face_mask_types: Optional[List[FaceMaskType]] = ['box', 'region', 'occlusion']
        self.face_mask_blur: Optional[float] = 0.3
        self.face_mask_padding: Optional[Padding] = (0, 0, 0, 0)
        self.face_mask_regions: Optional[List[FaceMaskRegion]] = face_mask_regions
        # frame extraction
        self.trim_frame_start: Optional[int] = None
        self.trim_frame_end: Optional[int] = None
        self.temp_frame_format: Optional[TempFrameFormat] = 'png'
        self.keep_temp: Optional[bool] = False
        # output creation
        self.output_image_quality: Optional[int] = 60
        self.output_image_resolution: Optional[str] = None
        self.output_video_encoder: Optional[OutputVideoEncoder] = 'libx264'
        self.output_video_preset: Optional[OutputVideoPreset] = 'veryfast'
        self.output_video_quality: Optional[int] = 60
        self.output_video_resolution: None
        self.output_video_fps: Optional[str] = None
        self.skip_audio: Optional[bool] = False
        # frame processors
        self.frame_processors: List[str] = ["face_swapper"]
        # uis
        self.ui_layouts: List[str] = ["default"]

        # Custom elements for AUTO extension
        self.mask_disabled_times: Optional[List[int]] = [0]
        self.mask_enabled_times: Optional[List[int]] = []
        self.reference_face_dict: Optional[dict] = {}
        self.reference_face_dict_2: Optional[dict] = {}
        self.restricted_path: Optional[str] = None
        self.source_paths_2: Optional[List[str]] = None
        self.sync_video_lip: Optional[bool] = False

    def compare(self, other):
        # Compare all of the values in this instance to another instance, excluding self.id
        # Return True if all values match, False otherwise
        if not isinstance(other, JobParams):
            return False
        for key in self.__dict__:
            if key != "id":
                if self.__dict__[key] != other.__dict__[key]:
                    return False
        return True

    def to_json(self):
        # Method to convert instance to JSON serializable dictionary
        out_params = {}
        for key in self.__dict__:
            if key.startswith("_"):
                continue
            value = self.__dict__[key]
            # Ensure that the value is JSON serializable
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                if key == "reference_face_dict" or key == "reference_face_dict_2":
                    continue
                out_params[key] = self.__dict__[key]

        return json.dumps(out_params, indent=4)

    def to_dict(self):
        # Method to convert instance to JSON serializable dictionary
        out_params = {}
        for key in self.__dict__:
            if key.startswith("_"):
                continue
            out_params[key] = self.__dict__[key]
        return out_params

    @classmethod
    def from_json(cls, json_str):
        # Method to create an instance from a JSON string
        json_dict = json.loads(json_str)
        params = cls()
        params.__dict__.update(json_dict)
        return params

    @classmethod
    def from_dict(cls, json_dict):
        # Method to create an instance from a JSON string
        params = cls()
        params.__dict__.update(json_dict)
        return params

    @classmethod
    def from_globals(cls):
        # Method to create an instance from the global parameters
        from facefusion.globals import __dict__ as globals_dict
        params = cls()
        params.__dict__.update(globals_dict)
        return params
