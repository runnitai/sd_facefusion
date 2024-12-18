from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import List, Any

from facefusion import logger, state_manager
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.filesystem import resolve_relative_path
from facefusion.typing import ModelSet


class BaseProcessor(ABC):
    """
    Abstract Base Processor for FaceFusion.
    Defines common attributes and methods for all processors.
    """
    MODEL_SET: ModelSet = None
    model_key: str = None
    __instance = None

    def __init__(self):
        self.inference_pool = None
        self.model_path = "../.assets/models"

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(BaseProcessor, cls).__new__(cls)
            cls.__instance.model_path = resolve_relative_path(cls.__instance.model_path)
            cls.__instance.MODEL_SET = cls.MODEL_SET
            cls.__instance.model_key = cls.model_key
        return cls.__instance

    @abstractmethod
    def register_args(self, program: ArgumentParser) -> None:
        """
        Register processor-specific arguments.
        """
        pass

    @abstractmethod
    def apply_args(self, args: Any, apply_state_item: Any) -> None:
        """
        Apply arguments to the processor.
        """
        pass

    @abstractmethod
    def get_inference_pool(self, model_context: str, model_sources: dict) -> None:
        """
        Retrieve or initialize the inference pool for the processor.
        """
        pass

    @abstractmethod
    def clear_inference_pool(self) -> None:
        """
        Clear the inference pool for memory management.
        """
        pass

    def pre_check(self) -> bool:
        """
        Perform pre-checks before processing.
        """
        download_directory_path = resolve_relative_path(self.model_path)
        model_hashes = self.get_model_options().get('hashes')
        model_sources = self.get_model_options().get('sources')

        return conditional_download_hashes(download_directory_path, model_hashes) and conditional_download_sources(
            download_directory_path, model_sources)

    @abstractmethod
    def pre_process(self, mode: str) -> bool:
        """
        Perform pre-processing steps based on the processing mode.
        """
        pass

    @abstractmethod
    def process_frame(self, inputs: dict) -> Any:
        """
        Process a single frame with specific inputs.
        """
        pass

    @abstractmethod
    def post_process(self) -> None:
        """
        Perform any cleanup or finalization steps after processing.
        """
        pass

    @abstractmethod
    def process_frames(self, queue_payloads: List[dict]) -> List[tuple]:
        """
        Process multiple frames.
        """
        pass

    @abstractmethod
    def process_image(self, target_path: str, output_path: str) -> None:
        """
        Process a single image.
        """
        pass

    def get_model_options(self) -> dict:
        """
        Get the model options for the processor.
        """
        model_choice = None
        try:
            model_choice = state_manager.get_item(self.model_key)
        except KeyError:
            print(f"Model choice not found for {self.model_key}.")
            pass
        model_options = self.MODEL_SET.get(model_choice)
        return model_options

    def process_video(self, temp_frame_paths: List[str]) -> None:
        """
        Process a video by handling its frames using multi-processing.
        """
        from facefusion.processors.core import multi_process_frames
        multi_process_frames(temp_frame_paths, self.process_frames)
        logger.info(f"Processed {len(temp_frame_paths)} frames successfully.", __name__)
