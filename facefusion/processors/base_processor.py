import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import List, Any, Tuple, Callable

from facefusion import logger, state_manager, inference_manager
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.filesystem import resolve_relative_path
from facefusion.typing import ModelSet, InferencePool, QueuePayload, VisionFrame, ProcessMode
from facefusion.workers.core import clear_worker_modules


class BaseProcessor(ABC):
    """
    Abstract Base Processor for FaceFusion.
    Defines common attributes and methods for all processors.
    """
    MODEL_SET: ModelSet = None
    model_key: str = None
    is_face_processor: bool = True
    priority = 1000
    model_path: str = "../.assets/models"
    default_model = None
    display_name: str = None
    context_name: str = None
    preload: bool = False
    preferred_provider: str = "default"
    __instances = {}

    def __init__(self):
        if self.MODEL_SET is None or self.model_key is None:
            raise ValueError("MODEL_SET and model_key must be defined in the child class.")
        self.inference_pool = None
        self.model_path = "../.assets/models"

    def __new__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super(BaseProcessor, cls).__new__(cls, *args, **kwargs)
            cls.__instances[cls].model_path = resolve_relative_path(cls.__instances[cls].model_path)
            class_name = cls.__name__
            # Handle CamelCase or snake_case to Title Case
            cls.__instances[cls].display_name = ' '.join(
                word.capitalize() for word in re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).split('_')
            )
            cls.__instances[cls].context_name = class_name
            cls.__instances[cls].model_key = cls.__instances[cls].model_key or f"{class_name}_model"
        return cls.__instances[cls]

    @abstractmethod
    def register_args(self, program: ArgumentParser) -> None:
        """Register processor-specific arguments."""
        pass

    @abstractmethod
    def apply_args(self, args: Any, apply_state_item: Callable) -> None:
        """Apply arguments to the processor."""
        pass

    @abstractmethod
    def pre_process(self, mode: ProcessMode) -> bool:
        """Perform pre-processing steps based on the processing mode."""
        pass

    @abstractmethod
    def process_frame(self, inputs: dict) -> VisionFrame:
        """Process a single frame with specific inputs."""
        pass

    def post_process(self) -> None:
        if state_manager.get_item("video_memory_strategy") in ["strict", "moderate"]:
            self.clear_inference_pool()
        if state_manager.get_item("video_memory_strategy") == "strict":
            clear_worker_modules()

    @abstractmethod
    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        """Process multiple frames."""
        pass

    @abstractmethod
    def process_image(self, target_path: str, output_path: str) -> None:
        """Process a single image."""
        pass

    def pre_check(self) -> bool:
        """
        Perform pre-checks before processing.
        """
        download_directory_path = resolve_relative_path(self.model_path)
        model_hashes = self.get_model_options().get('hashes', {})
        model_sources = self.get_model_options().get('sources', {})

        downloaded = (
                conditional_download_hashes(download_directory_path, model_hashes) and
                conditional_download_sources(download_directory_path, model_sources)
        )

        return downloaded

    def pre_load(self):
        if self.pre_check() and self.preload and not self.inference_pool:
            logger.debug(f"Preloaded: {self.display_name}", self.context_name)
            self.inference_pool = self.get_inference_pool()

    def set_inference_pool(self):
        """Sets the inference pool."""
        self.inference_pool = self.get_inference_pool()

    def get_inference_pool(self) -> InferencePool:
        if not self.get_model_options():
            if not self.set_model_options():
                raise ValueError(f"Model options not found for {self.model_key}.")
        model_sources = self.get_model_options().get('sources', [])
        model_context = self.context_name + '.' + (state_manager.get_item(self.model_key) or "default_key")
        self.inference_pool = inference_manager.get_inference_pool(model_context, model_sources)
        return self.inference_pool

    def clear_inference_pool(self) -> None:
        model_context = self.context_name + '.' + (state_manager.get_item(self.model_key) or "default_key")
        print(f"Clearing inference pool for {model_context}")
        inference_manager.clear_inference_pool(model_context)
        self.inference_pool = None

    def set_model_options(self):
        model_choice = state_manager.get_item(self.model_key)
        if model_choice is None:
            if self.default_model is not None:
                model_choice = self.default_model
                state_manager.set_item(self.model_key, model_choice)
                logger.error(f"Model choice not found for {self.model_key}.", __name__)
                return True
            return False
        model_options = self.MODEL_SET.get(model_choice, False)
        return model_options is not False

    def get_model_options(self) -> dict:
        """
        Get the model options for the processor.
        """
        model_choice = state_manager.get_item(self.model_key)
        if model_choice is None and self.default_model is not None:
            model_choice = self.default_model
            state_manager.set_item(self.model_key, model_choice)
            logger.error(f"Model choice not found for {self.model_key}.", __name__)
            return {}
        return self.MODEL_SET.get(model_choice, {})

    def process_video(self, temp_frame_paths: List[str]) -> None:
        """
        Process a video by handling its frames using multi-processing.
        """
        from facefusion.processors.core import multi_process_frames
        multi_process_frames(temp_frame_paths, self.process_frames)
        logger.info(f"Processed {len(temp_frame_paths)} frames successfully.", __name__)

    # Optional hooks for additional behavior
    def before_process(self):
        """Hook called before processing starts."""
        pass

    def after_process(self):
        """Hook called after processing finishes."""
        pass

    def list_models(self) -> List[str]:
        """
        List available models for the processor.
        """
        # Return all keys in MODEL_SET where 'internal' is not True or not set
        models = sorted([key for key, value in self.MODEL_SET.items() if not value.get('internal')])
        return models

    def download_all_models(self) -> None:
        download_directory_path = self.model_path
        all_hashes = {}
        all_sources = {}
        for model in self.MODEL_SET:
            model_hashes = self.MODEL_SET[model].get('hashes')
            model_sources = self.MODEL_SET[model].get('sources')
            all_hashes[model] = model_hashes
            all_sources[model] = model_sources
        return conditional_download_hashes(download_directory_path, all_hashes) and conditional_download_sources(
            download_directory_path, all_sources)
