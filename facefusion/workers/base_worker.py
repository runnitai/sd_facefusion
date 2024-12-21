import re
from abc import ABC
from argparse import ArgumentParser
from typing import List, Any, Callable, Tuple

from facefusion import logger, state_manager, inference_manager
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.filesystem import resolve_relative_path
from facefusion.typing import ModelSet, InferencePool, DownloadSet, ModelOptions


class BaseWorker(ABC):
    """
    Abstract Base Processor for FaceFusion.
    Defines common attributes and methods for all processors.
    """
    MODEL_SET: ModelSet = None
    model_key: str = None  # When set, this will be used as the key to store the model choice in the state manager
    default_model: str = None  # When set, this will be used as the default model choice
    model_path: str = "../.assets/models"
    display_name: str = None
    context_name: str = None
    multi_model: bool = False
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
            cls.__instances[cls] = super(BaseWorker, cls).__new__(cls, *args, **kwargs)
            cls.__instances[cls].model_path = resolve_relative_path(cls.__instances[cls].model_path)
            class_name = cls.__name__
            # Handle CamelCase or snake_case to Title Case
            cls.__instances[cls].display_name = ' '.join(
                word.capitalize() for word in re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).split('_')
            )
            cls.__instances[cls].context_name = class_name
            cls.__instances[cls].model_key = cls.__instances[cls].model_key or f"{class_name}_model"
        return cls.__instances[cls]

    def register_args(self, program: ArgumentParser) -> None:
        """Register processor-specific arguments."""
        pass

    def apply_args(self, args: Any, apply_state_item: Callable) -> None:
        """Apply arguments to the processor."""
        pass

    def pre_check(self) -> bool:
        download_directory_path = resolve_relative_path('../.assets/models')
        if self.multi_model:
            model_hashes, model_sources = self.collect_model_downloads()
        else:
            model_hashes = self.get_model_options().get('hashes')
            model_sources = self.get_model_options().get('sources')

        downloaded = (conditional_download_hashes(download_directory_path, model_hashes) and
                      conditional_download_sources(download_directory_path, model_sources))
        if downloaded and self.preload and not self.inference_pool:
            logger.debug(f"Preloaded: {self.display_name}", self.context_name)
            self.inference_pool = self.get_inference_pool()
        return downloaded

    def get_inference_pool(self) -> InferencePool:
        if self.multi_model:
            _, model_sources = self.collect_model_downloads()
            model_context = f"{self.context_name}.{state_manager.get_item(self.model_key)}" if self.model_key else __name__
        else:
            model_sources = self.get_model_options().get('sources')
            model_context = self.context_name
        return inference_manager.get_inference_pool(model_context, model_sources)

    def clear_inference_pool(self) -> None:
        model_context = f"{self.context_name}.{state_manager.get_item(self.model_key)}" if self.model_key else self.context_name
        inference_manager.clear_inference_pool(model_context)

    def get_model_options(self) -> ModelOptions:
        return self.MODEL_SET.get(self.default_model)

    def collect_model_downloads(self) -> Tuple[DownloadSet, DownloadSet]:
        # Overridden in subclasses if multi-model requires specific logic.
        model_hashes = {k: v['hashes'][k] for k, v in self.MODEL_SET.items()}
        model_sources = {k: v['sources'][k] for k, v in self.MODEL_SET.items()}
        return model_hashes, model_sources

    def list_models(self) -> List[str]:
        """
        List available models for the processor.
        """
        # Return all keys in MODEL_SET where 'internal' is not True or not set
        return [key for key, value in self.MODEL_SET.items() if not value.get('internal')]

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
