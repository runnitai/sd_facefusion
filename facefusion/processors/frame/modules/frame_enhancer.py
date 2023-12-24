import os
import threading
from argparse import ArgumentParser
from typing import Any, List, Literal, Optional

import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

import facefusion.processors.frame.core as frame_processors
from facefusion import wording
from facefusion.cli_helper import create_metavar
from facefusion.content_analyser import clear_content_analyser
from facefusion.download import conditional_download, is_download_done
from facefusion.execution_helper import map_device
from facefusion.face_analyser import clear_face_analyser
from facefusion.ff_status import FFStatus, update_status
from facefusion.filesystem import is_file, resolve_relative_path
from facefusion.job_params import JobParams
from facefusion.processors.frame import choices as frame_processors_choices
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.typing import Face, FaceSet, Frame, Update_Process, ProcessMode, ModelSet, OptionsWithModel
from facefusion.vision import read_image, read_static_image, write_image
from modules.paths_internal import models_path

FRAME_PROCESSOR = None
JOB = JobParams()
THREAD_SEMAPHORE: threading.Semaphore = threading.Semaphore()
THREAD_LOCK: threading.Lock = threading.Lock()
EXECUTION_THREAD_COUNT = 1
NAME = __name__.upper()
MODELS : ModelSet =\
{
    'real_esrgan_x2plus':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/real_esrgan_x2plus.pth',
        'path': resolve_relative_path('../.assets/models/real_esrgan_x2plus.pth'),
        'scale': 2
    },
    'real_esrgan_x4plus':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/real_esrgan_x4plus.pth',
        'path': resolve_relative_path('../.assets/models/real_esrgan_x4plus.pth'),
        'scale': 4
    },
    'real_esrnet_x4plus':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/real_esrnet_x4plus.pth',
        'path': resolve_relative_path('../.assets/models/real_esrnet_x4plus.pth'),
        'scale': 4
    }
}
OPTIONS: Optional[OptionsWithModel] = None


def get_frame_processor() -> Any:
    global FRAME_PROCESSOR

    with THREAD_LOCK:
        if FRAME_PROCESSOR is None:
            model_path = get_options('model').get('path')
            model_scale = get_options('model').get('scale')
            FRAME_PROCESSOR = RealESRGANer(
                model_path=model_path,
                model=RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    scale=model_scale
                ),
                device=map_device(["CUDAExecutionProvider"]),
                scale=model_scale
            )
    return FRAME_PROCESSOR


def clear_frame_processor() -> None:
    global FRAME_PROCESSOR

    FRAME_PROCESSOR = None


def get_options(key: Literal['model']) -> Any:
    global OPTIONS

    if OPTIONS is None:
        OPTIONS = \
        {
            'model': MODELS[frame_processors_globals.frame_enhancer_model]
        }
    return OPTIONS.get(key)


def set_options(key: Literal['model'], value: Any) -> None:
    global OPTIONS

    OPTIONS[key] = value


def register_args(program: ArgumentParser) -> None:
    program.add_argument('--frame-enhancer-model', help = wording.get('frame_processor_model_help'), default = 'real_esrgan_x2plus', choices = frame_processors_choices.frame_enhancer_models)
    program.add_argument('--frame-enhancer-blend', help = wording.get('frame_processor_blend_help'), type = int, default = 80, choices = frame_processors_choices.frame_enhancer_blend_range, metavar = create_metavar(frame_processors_choices.frame_enhancer_blend_range))


def apply_args(program: ArgumentParser) -> None:
    args = program.parse_args()
    frame_processors_globals.frame_enhancer_model = args.frame_enhancer_model
    frame_processors_globals.frame_enhancer_blend = args.frame_enhancer_blend


def pre_check() -> bool:
    download_directory_path = os.path.join(models_path, 'facefusion')
    model_url = get_options('model').get('url')
    conditional_download(download_directory_path, [model_url])
    return True


def pre_process(mode: ProcessMode, job: JobParams) -> bool:
    global EXECUTION_THREAD_COUNT
    global JOB
    EXECUTION_THREAD_COUNT = job.execution_thread_count
    JOB = job
    model_url = get_options('model').get('url')
    model_path = get_options('model').get('path')
    if not is_download_done(model_url, model_path):
        update_status(wording.get('model_download_not_done') + wording.get('exclamation_mark'), NAME)
        return False
    elif not is_file(model_path):
        update_status(f"Could not find model at path {model_path}", NAME)
        return False
    if mode == 'output' and not job.output_path:
        update_status(wording.get('select_file_or_directory_output') + wording.get('exclamation_mark'), NAME)
        return False
    return True


def post_process() -> None:
    clear_frame_processor()
    clear_face_analyser()
    clear_content_analyser()
    read_static_image.cache_clear()


def enhance_frame(temp_frame: Frame) -> Frame:
    with THREAD_SEMAPHORE:
        paste_frame, _ = get_frame_processor().enhance(temp_frame)
        temp_frame = blend_frame(temp_frame, paste_frame)
    return temp_frame


def blend_frame(temp_frame: Frame, paste_frame: Frame) -> Frame:
    frame_enhancer_blend = 1 - (frame_processors_globals.frame_enhancer_blend / 100)
    paste_frame_height, paste_frame_width = paste_frame.shape[0:2]
    temp_frame = cv2.resize(temp_frame, (paste_frame_width, paste_frame_height))
    temp_frame = cv2.addWeighted(temp_frame, frame_enhancer_blend, paste_frame, 1 - frame_enhancer_blend, 0)
    return temp_frame


def get_reference_frame(source_face : Face, target_face : Face, temp_frame : Frame) -> Frame:
    pass


def process_frame(source_face : Face, reference_faces : FaceSet, temp_frame : Frame) -> Frame:
    return enhance_frame(temp_frame)


def process_frames(source_paths: str, temp_frame_paths: List[str], update_progress: Update_Process,
                   status: FFStatus) -> None:
    frame_count = 0
    for temp_frame_path in temp_frame_paths:
        if status.cancelled:
            return temp_frame_paths
        temp_frame = read_image(temp_frame_path)
        result_frame = process_frame(None, None, temp_frame)
        write_image(temp_frame_path, result_frame)
        update_progress()
        frame_count += 1
        if frame_count % 30 == 0:
            status.update_preview(temp_frame_path)
    return temp_frame_paths


def process_image(source_paths : List[str], target_path : str, output_path : str) -> None:
    target_frame = read_static_image(target_path)
    result = process_frame(None, None, target_frame)
    write_image(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str], status) -> None:
    frame_processors.multi_process_frames(None, temp_frame_paths, process_frames, status)
