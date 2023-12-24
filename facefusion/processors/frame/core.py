import importlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from types import ModuleType
from typing import Any, List

import psutil

import facefusion.globals
from facefusion import wording
from facefusion.execution_helper import encode_execution_providers
from facefusion.mytqdm import mytqdm
from facefusion.typing import Process_Frames

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_METHODS = \
    [
        'get_frame_processor',
        'clear_frame_processor',
        'get_options',
        'set_options',
        'register_args',
        'apply_args',
        'pre_check',
        'pre_process',
        'get_reference_frame',
        'process_frame',
        'process_frames',
        'process_image',
        'process_video',
        'post_process'
    ]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module('facefusion.processors.frame.modules.' + frame_processor)
        for method_name in FRAME_PROCESSORS_METHODS:
            if not hasattr(frame_processor_module, method_name):
                raise NotImplementedError
    except ModuleNotFoundError:
        print(wording.get('frame_processor_not_found').format(frame_processor=frame_processor))
        return None
    except NotImplementedError:
        print(wording.get('frame_processor_not_implemented').format(frame_processor=frame_processor))
        return None
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    return FRAME_PROCESSORS_MODULES


def clear_frame_processors_modules() -> None:
    global FRAME_PROCESSORS_MODULES
    modules = ["face_swapper", "face_enhancer", "frame_enhancer", "face_debugger"]
    for frame_processor_module in get_frame_processors_modules(modules):
        frame_processor_module.clear_frame_processor()
    FRAME_PROCESSORS_MODULES = []


def multi_process_frames(source_paths: List[str], temp_frame_paths: List[str], process_frames: Process_Frames,
                         state) -> None:
    with mytqdm(total=len(temp_frame_paths), desc=wording.get('processing'), unit='frame', state=state) as progress:
        progress.set_postfix(
            {
                'execution_providers': encode_execution_providers(facefusion.globals.execution_providers),
                'execution_thread_count': facefusion.globals.execution_thread_count,
                'execution_queue_count': facefusion.globals.execution_queue_count
            })

        queue_temp_frame_paths: Queue[str] = create_queue(temp_frame_paths)

        with ThreadPoolExecutor() as executor:  # max_workers is chosen automatically
            futures = []
            while not queue_temp_frame_paths.empty():
                if state.cancelled:
                    return

                payload_temp_frame_paths = [queue_temp_frame_paths.get()]  # Process one frame at a time
                future = executor.submit(process_frames, source_paths, payload_temp_frame_paths,
                                         lambda: update_progress(1, 1, progress), state)
                futures.append(future)

            for future_done in as_completed(futures):
                if state.cancelled:
                    return
                future_done.result()


def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue


def update_progress(thread_count, queue_count, progress: Any = None) -> None:
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
    progress.set_postfix(
        {
            'memory_usage': '{:.2f}'.format(memory_usage).zfill(5) + 'GB',
            'execution_providers': ["CUDAExecutionProvider"],
            'execution_thread_count': thread_count,
            'execution_queue_count': queue_count
        })
    progress.refresh()
    progress.update(1)


def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues
