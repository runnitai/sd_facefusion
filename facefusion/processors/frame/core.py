import importlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from types import ModuleType
from typing import Any, List

import facefusion.globals
from facefusion import logger, wording
from facefusion.execution import encode_execution_providers
from facefusion.ff_status import FFStatus
from facefusion.mytqdm import mytqdm as tqdm
from facefusion.typing import ProcessFrames, QueuePayload

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
        'post_check',
        'pre_process',
        'post_process',
        'get_reference_frame',
        'process_frame',
        'process_frames',
        'process_image',
        'process_video'
    ]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module('facefusion.processors.frame.modules.' + frame_processor)
        for method_name in FRAME_PROCESSORS_METHODS:
            if not hasattr(frame_processor_module, method_name):
                raise NotImplementedError
    except ModuleNotFoundError as exception:
        logger.error(wording.get('frame_processor_not_loaded').format(frame_processor=frame_processor),
                     __name__.upper())
        logger.debug(exception.msg, __name__.upper())
        return None
    except NotImplementedError:
        logger.error(wording.get('frame_processor_not_implemented').format(frame_processor=frame_processor),
                     __name__.upper())
        return None
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    # Priority list defining the order
    priority_order = ['face_swapper', 'lip_syncer', 'face_enhancer', 'frame_enhancer', 'face_debugger']

    # Sort the frame_processors list based on the priority_order
    ordered_processors = sorted(frame_processors,
                                key=lambda x: priority_order.index(x) if x in priority_order else len(priority_order))

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in ordered_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)

    return FRAME_PROCESSORS_MODULES


def clear_frame_processors_modules() -> None:
    global FRAME_PROCESSORS_MODULES

    for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
        frame_processor_module.clear_frame_processor()
    FRAME_PROCESSORS_MODULES = []


# def multi_process_frames(source_paths: List[str], source_paths_2: List[str], temp_frame_paths: List[str], process_frames: ProcessFrames) -> None:
#     queue_payloads = create_queue_payloads(temp_frame_paths)
#     with tqdm(total=len(queue_payloads), desc=wording.get('processing'), unit='frame', ascii=' =',
#               disable=facefusion.globals.log_level in ['warn', 'error']) as progress:
#         progress.set_postfix(
#             {
#                 'execution_providers': encode_execution_providers(facefusion.globals.execution_providers),
#                 'execution_thread_count': facefusion.globals.execution_thread_count,
#                 'execution_queue_count': facefusion.globals.execution_queue_count
#             })
#         status = FFStatus()
#
#         def update_progress(preview_image: str = None) -> None:
#             progress.update()
#             if preview_image is not None:
#                 current_step = status.job_current
#                 if current_step % 30 == 0 or current_step == status.job_total:
#                     status.preview_image = preview_image
#
#         with ThreadPoolExecutor() as executor:
#             futures = []
#             queue: Queue[QueuePayload] = create_queue(queue_payloads)
#             while not queue.empty():
#                 future = executor.submit(process_frames, source_paths, source_paths_2,
#                                          pick_queue(queue, 1),
#                                          update_progress)
#                 futures.append(future)
#             for future_done in as_completed(futures):
#                 future_done.result()
def multi_process_frames(source_paths: List[str], source_paths_2: List[str], temp_frame_paths: List[str],
                         process_frames: ProcessFrames) -> None:
    queue_payloads = create_queue_payloads(temp_frame_paths)
    batch_size = facefusion.globals.batch_size or 4  # Default to 4 frames per batch
    max_workers = max(facefusion.globals.execution_thread_count, 4)  # Ensure sufficient threads

    with tqdm(total=len(queue_payloads), desc=wording.get('processing'), unit='frame', ascii=' =',
              disable=facefusion.globals.log_level in ['warn', 'error']) as progress:
        progress.set_postfix(
            {
                'execution_providers': encode_execution_providers(facefusion.globals.execution_providers),
                'execution_thread_count': max_workers,
                'execution_queue_count': facefusion.globals.execution_queue_count
            })
        status = FFStatus()
        frame_counter = 0

        def update_progress(preview_image: str = None) -> None:
            nonlocal frame_counter
            frame_counter += 1
            if frame_counter % 10 == 0 or frame_counter == status.job_total:
                progress.update(10)
            if preview_image is not None and frame_counter % 30 == 0:
                status.preview_image = preview_image

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            queue: Queue[QueuePayload] = create_queue(queue_payloads)
            while not queue.empty():
                batch = pick_queue(queue, batch_size)  # Fetch batches of frames
                future = executor.submit(process_frames, source_paths, source_paths_2, batch, update_progress)
                futures.append(future)

            for future_done in as_completed(futures):  # Wait for batch completion
                try:
                    future_done.result()
                except Exception as e:
                    print(f"Error processing batch: {e}")


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
    queue_payloads = []
    temp_frame_paths = sorted(temp_frame_paths, key=os.path.basename)

    for frame_number, frame_path in enumerate(temp_frame_paths):
        frame_payload: QueuePayload = \
            {
                'frame_number': frame_number,
                'frame_path': frame_path
            }
        queue_payloads.append(frame_payload)
    return queue_payloads
