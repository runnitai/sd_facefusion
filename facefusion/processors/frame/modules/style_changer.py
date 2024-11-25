import argparse
import os
import threading
from typing import Any, Literal, Optional, List

import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import facefusion.globals
import facefusion.processors.frame.core as frame_processors
from facefusion import config, logger, wording, process_manager
from facefusion.filesystem import is_image, is_video
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.processors.frame.typings import StyleChangerInputs
from facefusion.typing import ProcessMode, OptionsWithModel, VisionFrame, QueuePayload, UpdateProcess
from facefusion.vision import read_image, write_image, read_static_image

FRAME_PROCESSOR = None
THREAD_LOCK: threading.Lock = threading.Lock()
NAME = __name__.upper()

# Available models
MODEL_MAPPING = {
    "anime": "damo/cv_unet_person-image-cartoon_compound-models",
    "3d": "damo/cv_unet_person-image-cartoon-3d_compound-models",
    "handdrawn": "damo/cv_unet_person-image-cartoon-handdrawn_compound-models",
    "sketch": "damo/cv_unet_person-image-cartoon-sketch_compound-models",
    "artstyle": "damo/cv_unet_person-image-cartoon-artstyle_compound-models",
    "design": "damo/cv_unet_person-image-cartoon-sd-design_compound-models",
    "illustration": "damo/cv_unet_person-image-cartoon-sd-illustration_compound-models"
}
OPTIONS: Optional[OptionsWithModel] = None


def get_frame_processor() -> Any:
    global FRAME_PROCESSOR

    with THREAD_LOCK:
        if FRAME_PROCESSOR is None:
            model_name = get_options('model')
            FRAME_PROCESSOR = pipeline(Tasks.image_portrait_stylization, model=model_name)
    return FRAME_PROCESSOR


def clear_frame_processor() -> None:
    global FRAME_PROCESSOR
    FRAME_PROCESSOR = None


def get_options(key: Literal['model']) -> Any:
    global OPTIONS

    if OPTIONS is None:
        OPTIONS = \
            {
                'model': MODEL_MAPPING[frame_processors_globals.style_changer_model],
                'target': frame_processors_globals.style_changer_target
            }
    return OPTIONS.get(key)


def set_options(key: Literal['model'], value: Any) -> None:
    global OPTIONS
    OPTIONS[key] = value


# Register command-line arguments
def register_args(program: argparse.ArgumentParser) -> None:
    program.add_argument(
        '--style-changer-model',
        help=wording.get('help.style_changer_model'),
        default=config.get_str_value('style_changer_model', 'anime'),
        choices=list(MODEL_MAPPING.keys())
    )
    program.add_argument(
        '--style-changer-target',
        help=wording.get('help.style_changer_target'),
        default=config.get_str_value('style_changer_target', 'target'),
        choices=['source', 'target']
    )


# Apply command-line arguments
def apply_args(program: argparse.ArgumentParser) -> None:
    args = program.parse_args()
    facefusion.globals.style_changer_model = args.style_changer_model
    facefusion.globals.style_changer_target = args.style_changer_target


# Pre-check before processing
def pre_check() -> bool:
    model_name = facefusion.globals.style_changer_model
    if model_name not in MODEL_MAPPING:
        logger.error(f"Model '{model_name}' not found in available models.", "STYLE_CHANGER")
        return False
    # Todo: Figure out how to download the model and store it locally
    # model_path = resolve_relative_path(f'../.assets/models/{model_name}.onnx')
    # if not is_file(model_path):
    #     logger.error(f"Model file for '{model_name}' is not present.", "STYLE_CHANGER")
    #     return False
    return True


# Post-check after setup (not used here, but included for completeness)
def post_check() -> bool:
    return True


def pre_process(mode: ProcessMode) -> bool:
    if mode in ['output', 'preview'] and not is_image(facefusion.globals.target_path) and not is_video(
            facefusion.globals.target_path):
        logger.error(wording.get('select_image_or_video_target') + wording.get('exclamation_mark'), NAME)
        return False
    if mode == 'output' and not facefusion.globals.output_path:
        logger.error(wording.get('select_file_or_directory_output') + wording.get('exclamation_mark'), NAME)
        return False
    return True


def change_style(temp_vision_frame: VisionFrame) -> VisionFrame:
    frame_processor = get_frame_processor()
    result = frame_processor(temp_vision_frame)
    return result[OutputKeys.OUTPUT_IMG]


# Process an image
def process_src_image(input_path: str, style: str):
    # Output path is the input path, with _style_{style} appended to the file name
    input_file, input_extension = os.path.splitext(input_path)
    output_path = f"{input_file}_style_{style}{input_extension}"
    model_name = MODEL_MAPPING[style]
    img_cartoon = pipeline(Tasks.image_portrait_stylization, model=model_name)
    result = img_cartoon(input_path)
    cv2.imwrite(output_path, result[OutputKeys.OUTPUT_IMG])
    print(f"Image processing complete. Output saved to {output_path}.")
    return output_path


# Post-process results (placeholder for cleanup actions)
def post_process() -> None:
    pass


def process_frame(inputs: StyleChangerInputs) -> VisionFrame:
    if get_options('target') == 'source':
        return inputs.get('target_vision_frame')
    target_vision_frame = inputs.get('target_vision_frame')
    return change_style(target_vision_frame)


def process_frames(source_paths: List[str], source_paths_2: List[str], queue_payloads: List[QueuePayload],
                   update_progress: UpdateProcess) -> None:
    if get_options('target') == 'source':
        return
    for queue_payload in process_manager.manage(queue_payloads):
        target_vision_path = queue_payload['frame_path']
        target_vision_frame = read_image(target_vision_path)
        output_vision_frame = process_frame(
            {
                'target_vision_frame': target_vision_frame
            })
        write_image(target_vision_path, output_vision_frame)
        update_progress(target_vision_path)


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
    if get_options('target') == 'source':
        return
    target_vision_frame = read_static_image(target_path)
    output_vision_frame = process_frame(
        {
            'target_vision_frame': target_vision_frame
        })
    write_image(output_path, output_vision_frame)


def process_video(source_paths: List[str], source_paths_2: List[str], temp_frame_paths: List[str]) -> None:
    frame_processors.multi_process_frames(None, None, temp_frame_paths, process_frames)
