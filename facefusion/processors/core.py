import importlib
import inspect
import os
import pkgutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Any, List, Dict

import facefusion.globals
from facefusion import logger, wording, state_manager
from facefusion.face_analyser import get_avg_faces
from facefusion.ff_status import FFStatus
from facefusion.mytqdm import mytqdm as tqdm
from facefusion.processors.classes.base_processor import BaseProcessor
from facefusion.typing import ProcessFrames, QueuePayload

PROCESSORS_METHODS = \
    [
        'get_inference_pool',
        'clear_inference_pool',
        'register_args',
        'apply_args',
        'pre_check',
        'pre_process',
        'post_process',
        'process_frame',
        'process_frames',
        'process_image',
        'process_video'
    ]

PROCESSOR_INSTANCES: Dict[str, BaseProcessor] = {}


def load_processor_module(processor: str) -> Any:
    processors = [processor]
    processor_module = None
    try:
        processor_modules = get_processors_modules(processors)
        processor_module = processor_modules[0]
    except Exception as e:
        logger.error(f"Failed to load processor module {processor}: {e}", __name__)
    return processor_module


def get_processors_modules(frame_processors: List[str] = None) -> List[BaseProcessor]:
    """
    Discover all subclasses of BaseProcessor across the facefusion.processors package
    and instantiate them as singletons.
    """
    global PROCESSOR_INSTANCES
    from facefusion.processors.classes.base_processor import BaseProcessor
    if frame_processors is None:
        frame_processors = []
    # Iterate through all submodules in the facefusion.processors package
    for loader, module_name, is_pkg in pkgutil.walk_packages(path=facefusion.processors.classes.__path__,
                                                             prefix="facefusion.processors.classes."):
        try:
            # Dynamically import the module
            module = importlib.import_module(module_name)
        except Exception as e:
            print(f"Failed to import {module_name}: {e}")
            continue  # Skip modules that can't be imported

        # Find all subclasses of BaseProcessor in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            try:
                if issubclass(obj, BaseProcessor) and obj is not BaseProcessor:
                    # Avoid duplicate instantiation and check if already loaded
                    if name not in PROCESSOR_INSTANCES:
                        instance = obj()
                        PROCESSOR_INSTANCES[name] = instance
                        print(f"Loaded processor: {name}")
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
    sorted_processors.sort(key=lambda x: x.priority)
    return sorted_processors


def list_processors(frame_processors: List[str] = None) -> List[str]:
    all_processors = get_processors_modules(frame_processors)
    return [processor.display_name for processor in all_processors]


def clear_processors_modules(processors: List[str]) -> None:
    for processor in get_processors_modules(processors):
        processor.clear_inference_pool()


def multi_process_frames(temp_frame_paths: List[str], process_frames: ProcessFrames) -> None:
    queue_payloads = create_queue_payloads(temp_frame_paths)

    with tqdm(total=len(queue_payloads), desc=wording.get('processing'), unit='frame', ascii=' =',
              disable=state_manager.get_item('log_level') in ['warn', 'error']) as progress:
        progress.set_postfix(
            {
                'execution_providers': state_manager.get_item('execution_providers'),
                'execution_thread_count': state_manager.get_item('execution_thread_count'),
                'execution_queue_count': state_manager.get_item('execution_queue_count')
            })
        status = FFStatus()

        def update_progress(preview_image: str = None) -> None:
            progress.update()
            if preview_image is not None:
                current_step = status.job_current
                if current_step % 30 == 0 or current_step == status.job_total:
                    status.preview_image = preview_image

        with ThreadPoolExecutor(max_workers=state_manager.get_item('execution_thread_count')) as executor:
            futures = []
            queue: Queue[QueuePayload] = create_queue(queue_payloads)
            while not queue.empty():
                future = executor.submit(process_frames, pick_queue(queue, 1))
                futures.append(future)
            for future_done in as_completed(futures):
                try:
                    results = future_done.result()
                    for result in results:
                        if isinstance(result, tuple):
                            frame_number, processed_path = result
                            if frame_number % 10 == 0 or frame_number == status.job_total:
                                update_progress(processed_path)
                            else:
                                update_progress()
                        else:
                            print("Error: ", result)

                except Exception as e:
                    print("Error: ", e)
                    traceback.print_exc()
                    pass


def create_queue(queue_payloads: List[QueuePayload]) -> Queue[QueuePayload]:
    queue: Queue[QueuePayload] = Queue()
    for queue_payload in queue_payloads:
        queue.put(queue_payload)
    return queue


def pick_queue(queue: Queue[QueuePayload], queue_per_future: int) -> List[QueuePayload]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues


def create_queue_payloads(temp_frame_paths: List[str]) -> List[QueuePayload]:
    from facefusion.face_store import get_reference_faces

    queue_payloads = []
    source_face, source_face_2 = get_avg_faces()
    reference_faces, reference_faces_2 = (
        get_reference_faces() if 'reference' in facefusion.globals.face_selector_mode else (None, None))

    temp_frame_paths = sorted(temp_frame_paths, key=os.path.basename)

    for frame_number, frame_path in enumerate(temp_frame_paths):
        frame_payload: QueuePayload = \
            {
                'frame_number': frame_number,
                'frame_path': frame_path,
                'source_face': source_face,
                'source_face_2': source_face_2,
                'reference_faces': reference_faces,
                'reference_faces_2': reference_faces_2
            }
        queue_payloads.append(frame_payload)
    return queue_payloads
