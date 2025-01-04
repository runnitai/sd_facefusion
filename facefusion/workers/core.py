import importlib
import inspect
import pkgutil
from typing import List, Any, Dict

from facefusion import logger
from facefusion.workers import classes
from facefusion.workers.base_worker import BaseWorker

PROCESSOR_INSTANCES: Dict[str, BaseWorker] = {}


def load_worker_module(processor: str) -> Any:
    processor_module = None
    try:
        processor_modules = get_worker_modules()
        for processor_module in processor_modules:
            if processor_module.display_name == processor:
                return processor_module
    except Exception as e:
        logger.error(f"Failed to load processor module {processor}: {e}", __name__)
    return processor_module


def get_worker_modules() -> List[BaseWorker]:
    """
    Discover all subclasses of BaseWorker and return their instances.
    """
    worker_instances = {}

    # Discover and load all worker modules
    for loader, module_name, is_pkg in pkgutil.walk_packages(
            path=classes.__path__, prefix="facefusion.workers.classes."
    ):
        if not module_name.startswith("facefusion.workers.classes."):
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            logger.error(f"Failed to import {module_name}: {e}", __name__)
            continue

        for name, obj in inspect.getmembers(module):
            # Ensure obj is a valid class and check issubclass safely
            if isinstance(obj, type):
                try:
                    if issubclass(obj, BaseWorker) and obj is not BaseWorker:
                        if name not in worker_instances:
                            try:
                                instance = obj()
                                worker_instances[name] = instance
                            except Exception as e:
                                logger.error(f"Failed to instantiate worker {name}: {e}", __name__)
                except TypeError:
                    # Skip non-class objects or invalid subclass checks
                    continue

    return list(worker_instances.values())


def list_workers() -> List[str]:
    all_processors = get_worker_modules()
    return [processor.display_name for processor in all_processors]


def clear_worker_modules() -> None:
    for processor in get_worker_modules():
        processor.clear_inference_pool()
