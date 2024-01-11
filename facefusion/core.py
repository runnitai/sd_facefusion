import os
import traceback

from facefusion.content_analyser import analyse_image, analyse_video
from facefusion.execution_helper import decode_execution_providers
from facefusion.face_analyser import get_one_face, get_average_face

from facefusion.face_store import get_reference_faces, append_reference_face
from facefusion.ffmpeg import compress_image, extract_frames, merge_video, restore_audio
from facefusion.filesystem import is_image, is_video, create_temp, get_temp_frame_paths, clear_temp, move_temp, \
    list_module_names
from facefusion.normalizer import normalize_output_path, normalize_padding
from facefusion.vision import get_video_frame, read_image, detect_fps, read_static_images
from facefusion.ff_status import FFStatus
from facefusion.job_params import JobParams

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import warnings
import platform
import shutil
from argparse import ArgumentParser

import facefusion.globals
from facefusion import face_analyser, face_masker, content_analyser, metadata, logger, wording
from facefusion import wording, content_analyser
from facefusion.processors.frame.core import get_frame_processors_modules, load_frame_processor_module

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def apply_args(program: ArgumentParser) -> None:
    args = program.parse_args()
    # general
    facefusion.globals.source_paths = args.source_paths
    facefusion.globals.target_path = args.target_path
    facefusion.globals.output_path = normalize_output_path(facefusion.globals.source_paths,
                                                           facefusion.globals.target_path, args.output_path)
    # misc
    facefusion.globals.skip_download = args.skip_download
    facefusion.globals.headless = args.headless
    facefusion.globals.log_level = args.log_level
    # execution
    facefusion.globals.execution_providers = decode_execution_providers(args.execution_providers)
    facefusion.globals.execution_thread_count = args.execution_thread_count
    facefusion.globals.execution_queue_count = args.execution_queue_count
    facefusion.globals.max_memory = args.max_memory
    # face analyser
    facefusion.globals.face_analyser_order = args.face_analyser_order
    facefusion.globals.face_analyser_age = args.face_analyser_age
    facefusion.globals.face_analyser_gender = args.face_analyser_gender
    facefusion.globals.face_detector_model = args.face_detector_model
    facefusion.globals.face_detector_size = args.face_detector_size
    facefusion.globals.face_detector_score = args.face_detector_score
    # face selector
    facefusion.globals.face_selector_mode = args.face_selector_mode
    facefusion.globals.reference_face_position = args.reference_face_position
    facefusion.globals.reference_face_distance = args.reference_face_distance
    facefusion.globals.reference_frame_number = args.reference_frame_number
    # face mask
    facefusion.globals.face_mask_types = args.face_mask_types
    facefusion.globals.face_mask_blur = args.face_mask_blur
    facefusion.globals.face_mask_padding = normalize_padding(args.face_mask_padding)
    facefusion.globals.face_mask_regions = args.face_mask_regions
    # frame extraction
    facefusion.globals.trim_frame_start = args.trim_frame_start
    facefusion.globals.trim_frame_end = args.trim_frame_end
    facefusion.globals.temp_frame_format = args.temp_frame_format
    facefusion.globals.temp_frame_quality = args.temp_frame_quality
    facefusion.globals.keep_temp = args.keep_temp
    # output creation
    facefusion.globals.output_image_quality = args.output_image_quality
    facefusion.globals.output_video_encoder = args.output_video_encoder
    facefusion.globals.output_video_preset = args.output_video_preset
    facefusion.globals.output_video_quality = args.output_video_quality
    facefusion.globals.keep_fps = args.keep_fps
    facefusion.globals.skip_audio = args.skip_audio
    # frame processors
    available_frame_processors = list_directory('facefusion/processors/frame/modules')
    facefusion.globals.frame_processors = args.frame_processors
    for frame_processor in available_frame_processors:
        frame_processor_module = load_frame_processor_module(frame_processor)
        frame_processor_module.apply_args(program)
    # uis
    facefusion.globals.ui_layouts = args.ui_layouts


def run(program: ArgumentParser) -> None:
    apply_args(program)
    logger.init(facefusion.globals.log_level)
    limit_resources()
    if not pre_check() or not content_analyser.pre_check() or not face_analyser.pre_check() or not face_masker.pre_check():
        return
    for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
        if not frame_processor_module.pre_check():
            return
    if facefusion.globals.headless:
        conditional_process()
    else:
        import facefusion.uis.core as ui
        for ui_layout in ui.get_ui_layouts_modules(facefusion.globals.ui_layouts):
            if not ui_layout.pre_check():
                return
        ui.launch()


def destroy() -> None:
    if facefusion.globals.target_path:
        clear_temp()
    sys.exit()


def limit_resources() -> None:
    if facefusion.globals.max_memory:
        memory = facefusion.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = facefusion.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
    status = FFStatus()
    if sys.version_info < (3, 9):
        status.update(wording.get('python_not_supported').format(version='3.9'))
        return False
    if not shutil.which('ffmpeg'):
        status.update(wording.get('ffmpeg_not_installed'))
        return False
    return True


def conditional_process(status: FFStatus, job: JobParams) -> None:
    conditional_append_reference_faces(job)
    for frame_processor_module in get_frame_processors_modules(job.frame_processors):
        if not frame_processor_module.pre_process('output'):
            return
    target_path = job.target_path
    print(f"Processing {target_path}")
    try:
        if is_image(target_path):
            process_image(status, job)
        if is_video(target_path):
            process_video(status, job)
    except Exception as e:
        print(f"Exception Processing: {e}")
        traceback.print_exc()


def conditional_append_reference_faces(job: JobParams = None) -> None:
    if not job:
        job = JobParams().from_dict(facefusion.globals.__dict__)
    if 'reference' in job.face_selector_mode and not get_reference_faces():
        source_frames = read_static_images(job.source_paths)
        source_face = get_average_face(source_frames)
        if is_video(job.target_path):
            reference_frame = get_video_frame(job.target_path, job.reference_frame_number)
        else:
            reference_frame = read_image(job.target_path)
        reference_face = get_one_face(reference_frame, job.reference_face_position)
        append_reference_face('origin', reference_face)
        if source_face and reference_face:
            for frame_processor_module in get_frame_processors_modules(job.frame_processors):
                reference_frame = frame_processor_module.get_reference_frame(source_face, reference_face,
                                                                             reference_frame)
                reference_face = get_one_face(reference_frame, job.reference_face_position)
                append_reference_face(frame_processor_module.__name__, reference_face)


def process_image(status, job: JobParams) -> None:
    if analyse_image(job.target_path):
        status.update("Naughty naughty!!")
        status.cancel()
        return
    shutil.copy2(job.target_path, job.output_path)
    # process frame
    frame_processor_modules = get_frame_processors_modules(job.frame_processors)
    for frame_processor_module in frame_processor_modules:
        if status.cancelled:
            print("Interrupted")
            break
        status.update(f"{wording.get('processing')} {frame_processor_module.NAME}")
        frame_processor_module.process_image(job.source_paths, job.output_path, job.output_path)
        frame_processor_module.post_process()
        status.step()
    # compress image
    status.update(wording.get('compressing_image'))
    if status.cancelled:
        print("Interrupted")
        return
    if not compress_image(job.output_path):
        status_str = wording.get('compressing_image_failed')
        status.update(status_str)
    # validate image
    if is_image(job.target_path):
        status_str = wording.get('processing_image_succeed')
    else:
        status_str = wording.get('processing_image_failed')
    status.update(status_str)


def process_video(status, job) -> None:
    if analyse_video(job.target_path, job.trim_frame_start,
                     job.trim_frame_end):
        status.update("Naughty naughty!!")
        status.cancel()
        return
    status.update("Processing facefusion video.")
    fps = detect_fps(job.target_path) if job.keep_fps else 25.0
    # create temp
    create_temp(job.target_path)
    audio_path = None
    if job.source_speaker_path is not None and not job.skip_audio:
        if os.path.exists(job.source_speaker_path):
            status.update("Cloning audio...")
            # audio_path = clone_audio(job.source_speaker_path, job.target_path)
    else:
        audio_path = None

    # extract frames
    status.update(f"Extracting frames from {os.path.basename(job.target_path)}...")
    extract_frames(job.target_path, fps, status)
    status.step()
    # process frame
    temp_frame_paths = get_temp_frame_paths(job.target_path)
    if temp_frame_paths:
        for frame_processor_module in get_frame_processors_modules(job.frame_processors):
            if status.cancelled:
                print("Interrupted")
                clear_temp()
                return
            module_name = frame_processor_module.NAME
            # Split the module name by "." and select the last bit
            module_name = module_name.split(".")[-1]
            # Replace "_" with spaces and title case it
            module_name = module_name.replace("_", " ").title()
            status.update(f"Processing with {module_name}")
            frame_processor_module.process_video(job.source_paths, temp_frame_paths, status)
            frame_processor_module.post_process()
    else:
        status.update(wording.get('temp_frames_not_found'))
        return
    # merge video
    if status.cancelled:
        print("Interrupted")
        clear_temp()
        return
    status.update(f"Merging video to {job.output_path} ({fps} fps)")
    status.step()
    if not merge_video(job.target_path, fps, status):
        status.update(wording.get('merging_video_failed'))
    # handle audio
    if job.skip_audio:
        status.update(wording.get('skipping_audio'))
        move_temp(job.target_path, job.output_path)
    else:
        if status.cancelled:
            print("Interrupted")
            clear_temp()
            return
        try:
            status.update(wording.get('restoring_audio'))
            status.step()
            if not restore_audio(job.target_path, job.output_path, audio_path, status):
                print("Failed to restore audio")
                status.update(wording.get('restoring_audio_failed'))
                move_temp(job.target_path, job.output_path)
        except:
            print("Failed to restore audio")
            status.update(wording.get('restoring_audio_failed'))
    # clear temp
    status.update(wording.get('clearing_temp'))
    clear_temp()
    # validate video
    if is_video(job.target_path):
        status.update(wording.get('processing_video_succeed'))
    else:
        status.update(wording.get('processing_video_failed'))
