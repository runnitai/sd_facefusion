import importlib
import inspect
import pkgutil
from typing import List, Any, Dict

from facefusion import logger
from facefusion.workers import classes
from facefusion.workers.base_worker import BaseWorker

PROCESSOR_INSTANCES: Dict[str, BaseWorker] = {}


def load_worker_module(processor: str) -> Any:
    processors = [processor]
    processor_module = None
    try:
        processor_modules = get_worker_modules(processors)
        processor_module = processor_modules[0]
    except Exception as e:
        logger.error(f"Failed to load processor module {processor}: {e}", __name__)
    return processor_module


def get_worker_modules(frame_processors: List[str] = None) -> List[BaseWorker]:
    """
    Discover all subclasses of BaseProcessor across the facefusion.processors package
    and instantiate them as singletons.
    """
    global PROCESSOR_INSTANCES
    if frame_processors is None:
        frame_processors = []
    # Iterate through all submodules in the facefusion.processors package
    classes_path = classes.__path__
    for loader, module_name, is_pkg in pkgutil.walk_packages(path=classes_path,
                                                             prefix="facefusion.workers.classes."):
        try:
            # Dynamically import the module
            module = importlib.import_module(module_name)
        except Exception as e:
            print(f"Failed to import {module_name}: {e}")
            continue  # Skip modules that can't be imported

        # Find all subclasses of BaseProcessor in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            try:
                if issubclass(obj, BaseWorker) and obj is not BaseWorker:
                    # Avoid duplicate instantiation and check if already loaded
                    if name not in PROCESSOR_INSTANCES:
                        instance = obj()
                        if not instance.pre_check():
                            logger.error(f"Failed to pre-check processor: {name}", __name__)
                        PROCESSOR_INSTANCES[name] = instance
                        logger.info(f"Loaded worker: {name}", __name__)
            except Exception as e:
                pass
    sorted_processors = []
    # Sort the processors based on their priority attribute
    title_processors = []
    for processor_string in frame_processors:
        if "_" in processor_string:
            processor_string = processor_string.replace("_", " ").title()
        title_processors.append(processor_string)
    frame_processors = title_processors
    for processor in PROCESSOR_INSTANCES.values():
        if len(frame_processors) == 0 or (processor.display_name in frame_processors):
            sorted_processors.append(processor)
    sorted_processors.sort(key=lambda x: x.display_name)
    return sorted_processors


def list_workers(frame_processors: List[str] = None) -> List[str]:
    all_processors = get_worker_modules(frame_processors)
    return [processor.display_name for processor in all_processors]


def clear_worker_modules(processors: List[str] = None) -> None:
    for processor in get_worker_modules(processors):
        processor.clear_inference_pool()
