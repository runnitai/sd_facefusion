import importlib
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from types import ModuleType
from typing import Any, List

import facefusion.globals
from facefusion import logger, wording, state_manager
from facefusion.face_analyser import get_avg_faces
from facefusion.ff_status import FFStatus
from facefusion.mytqdm import mytqdm as tqdm
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
        'get_reference_frame',
        'process_frame',
        'process_frames',
        'process_image',
        'process_video'
    ]


def load_processor_module(processor: str) -> Any:
    processor_module = None
    last_method_name = None
    try:
        processor_module = importlib.import_module('facefusion.processors.modules.' + processor)
        for method_name in PROCESSORS_METHODS:
            if not hasattr(processor_module, method_name):
                last_method_name = method_name
                raise NotImplementedError
    except ModuleNotFoundError as exception:
        logger.error(wording.get('processor_not_loaded').format(processor=processor), __name__)
        logger.debug(exception.msg, __name__)
        # hard_exit(1)
    except NotImplementedError:
        logger.error(wording.get('processor_not_implemented').format(processor=processor), __name__)
        print("Error: ", last_method_name)
        # hard_exit(1)
    return processor_module


def get_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    processor_modules = []

    # Priority list defining the order
    priority_order = ['face_swapper', 'age_modifier', 'lip_syncer', 'face_editor', 'expression_restorer', 'style_changer', 'face_enhancer', 'frame_enhancer', 'frame_colorizer', 'face_debugger']

    # Sort the frame_processors list based on the priority_order
    ordered_processors = sorted(frame_processors,
                                key=lambda x: priority_order.index(x) if x in priority_order else len(priority_order))

    for processor in ordered_processors:
        processor_module = load_processor_module(processor)
        processor_modules.append(processor_module)
    return processor_modules


def clear_processors_modules(processors: List[str]) -> None:
    for processor in processors:
        processor_module = load_processor_module(processor)
        processor_module.clear_inference_pool()


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
