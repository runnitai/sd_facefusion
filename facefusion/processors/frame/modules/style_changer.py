import argparse
import os
import threading
from typing import Any, Literal, Optional, List

import cv2
import numpy as np
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import facefusion.globals
import facefusion.processors.frame.core as frame_processors
from facefusion import config, logger, wording, process_manager
from facefusion.filesystem import is_image, is_video
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.processors.frame.typings import StyleChangerInputs
from facefusion.typing import ProcessMode, OptionsWithModel, VisionFrame, QueuePayload, UpdateProcess, Face
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
    "illustration": "damo/cv_unet_person-image-cartoon-sd-illustration_compound-models",
    "genshen": "lskhh/moran-cv_unet_person-image-cartoon-genshin_compound-models",
    "anime2": "lskhh/ty_cv_unet_person-image-cartoon-wz_compound-models"
}
OPTIONS: Optional[OptionsWithModel] = None


def get_frame_processor() -> Any:
    global FRAME_PROCESSOR

    with THREAD_LOCK:
        if FRAME_PROCESSOR is None:
            model = get_options('model')
            model_name = MODEL_MAPPING.get(model)
            print(f"Loading style changer model: {model_name}")
            FRAME_PROCESSOR = pipeline(Tasks.image_portrait_stylization, model=model_name)
    return FRAME_PROCESSOR


def clear_frame_processor() -> None:
    global FRAME_PROCESSOR
    if FRAME_PROCESSOR is not None:
        print("Deleting frame processor")
        del FRAME_PROCESSOR
        print("Frame processor deleted")
    FRAME_PROCESSOR = None


def get_options(key: Literal['model']) -> Any:
    global OPTIONS

    if OPTIONS is None:
        OPTIONS = \
            {
                'model': frame_processors_globals.style_changer_model,
                'target': frame_processors_globals.style_changer_target
            }
    return OPTIONS.get(key)


def set_options(key: Literal['model'], value: Any) -> None:
    global OPTIONS
    if not OPTIONS:
        OPTIONS = \
            {
                'model': MODEL_MAPPING[frame_processors_globals.style_changer_model],
                'target': frame_processors_globals.style_changer_target
            }
    if key == 'model' and OPTIONS.get(key) != value and FRAME_PROCESSOR:
        print("Clearing frame processor")
        clear_frame_processor()
        print("Frame processor cleared")
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
    print(f"Pre-check passed for model '{model_name}'.")
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

    # Extract the processed image
    img = result.get(OutputKeys.OUTPUT_IMG)

    # If the output image is None, return the input frame directly
    if img is None:
        print("No processed image returned; using input frame.")
        return temp_vision_frame

    # Ensure the image is in float format for processing
    if img.dtype not in [np.float32, np.float64]:
        img = img.astype(np.float32)

    # Check the range of the image
    img_min, img_max = np.min(img), np.max(img)

    # Normalize based on the range
    if img_min >= -1.0 and img_max <= 1.0:
        img = (img + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
    elif img_min >= 0.0 and img_max <= 1.0:
        pass  # Already in [0, 1], no normalization needed
    elif img_min >= 0.0 and img_max <= 255.0:
        img = img / 255.0  # Normalize to [0, 1]
    else:
        img = (img - img_min) / (img_max - img_min)  # Fallback normalization

    # Ensure final range is clipped to [0, 1]
    img = np.clip(img, 0, 1)

    # Convert to uint8 for visualization or output
    img = (img * 255).astype(np.uint8)

    # Resize the image to match the input frame dimensions if necessary
    input_h, input_w, _ = temp_vision_frame.shape
    if img.shape[:2] != (input_h, input_w):
        img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_AREA)

    return img


def get_reference_frame(source_face: Face, target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
    if get_options('target') == 'source':
        return temp_vision_frame
    return change_style(temp_vision_frame)


# Process an image
def process_src_image(input_path: str, style: str):
    # Output path is the input path, with _style_{style} appended to the file name
    input_file, input_extension = os.path.splitext(input_path)
    output_path = f"{input_file}_style_{style}{input_extension}"
    processor = get_frame_processor()
    result = processor(input_path)
    cv2.imwrite(output_path, result[OutputKeys.OUTPUT_IMG])
    print(f"Image processing complete. Output saved to {output_path}.")
    return output_path


# Post-process results (placeholder for cleanup actions)
def post_process() -> None:
    pass


def process_frame(inputs: StyleChangerInputs) -> VisionFrame:
    target_vision_frame = inputs.get('target_vision_frame')
    converted = change_style(target_vision_frame)
    return converted


def process_frames(source_paths: List[str], source_paths_2: List[str], queue_payloads: List[QueuePayload],
                   update_progress: UpdateProcess) -> None:
    if get_options('target') == 'source':
        print(f"Skipping processing for source target: {get_options('target')}")
        return
    for queue_payload in queue_payloads:
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
    print("Processing video frames with style changer.")
    frame_processors.multi_process_frames(None, None, temp_frame_paths, process_frames)
