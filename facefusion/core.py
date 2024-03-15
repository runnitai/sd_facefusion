import os

os.environ['OMP_NUM_THREADS'] = '1'
import shutil
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser, HelpFormatter
from asyncio import sleep

import numpy
import onnxruntime

import facefusion.globals
from facefusion import face_analyser, face_masker, logger, metadata, config
from facefusion import wording, content_analyser, choices
from facefusion.common_helper import get_first, create_metavar
from facefusion.content_analyser import analyse_image, analyse_video
from facefusion.execution import decode_execution_providers, encode_execution_providers
from facefusion.face_analyser import get_one_face, get_average_face
from facefusion.face_store import get_reference_faces, append_reference_face
from facefusion.ff_status import FFStatus
from facefusion.ffmpeg import compress_image, extract_frames, merge_video, restore_audio, replace_audio
from facefusion.filesystem import is_image, is_video, create_temp, get_temp_frame_paths, clear_temp, move_temp, \
    list_directory, filter_audio_paths
from facefusion.job_params import JobParams
from facefusion.memory import limit_system_memory
from facefusion.normalizer import normalize_output_path, normalize_padding, normalize_fps
from facefusion.processors.frame.core import get_frame_processors_modules, load_frame_processor_module
from facefusion.typing import Face
from facefusion.vision import get_video_frame, read_image, detect_video_fps, read_static_images, \
    create_video_resolutions, \
    detect_video_resolution, pack_resolution, detect_video_fps, detect_image_resolution, create_image_resolutions

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

onnxruntime.set_default_logger_severity(3)
warnings.filterwarnings('ignore', category=UserWarning, module='gradio')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def cli() -> None:
    program = ArgumentParser(formatter_class=lambda prog: HelpFormatter(prog, max_help_position=130), add_help=False)
    # general
    program.add_argument('-s', '--source', help=wording.get('help.source'), action='append', dest='source_paths',
                         default=config.get_str_list('general.source_paths'))
    program.add_argument('-t', '--target', help=wording.get('help.target'), dest='target_path',
                         default=config.get_str_value('general.target_path'))
    program.add_argument('-o', '--output', help=wording.get('help.output'), dest='output_path',
                         default=config.get_str_value('general.output_path'))
    program.add_argument('-v', '--version', version=metadata.get('name') + ' ' + metadata.get('version'),
                         action='version')
    # misc
    group_misc = program.add_argument_group('misc')
    group_misc.add_argument('--skip-download', help=wording.get('help.skip_download'), action='store_true',
                            default=config.get_bool_value('misc.skip_download'))
    group_misc.add_argument('--headless', help=wording.get('help.headless'), action='store_true',
                            default=config.get_bool_value('misc.headless'))
    group_misc.add_argument('--log-level', help=wording.get('help.log_level'),
                            default=config.get_str_value('misc.log_level', 'info'), choices=logger.get_log_levels())
    # execution
    execution_providers = encode_execution_providers(onnxruntime.get_available_providers())
    group_execution = program.add_argument_group('execution')
    group_execution.add_argument('--execution-providers', help=wording.get('help.execution_providers').format(
        choices=', '.join(execution_providers)), default=config.get_str_list('execution.execution_providers', 'cpu'),
                                 choices=execution_providers, nargs='+', metavar='EXECUTION_PROVIDERS')
    group_execution.add_argument('--execution-thread-count', help=wording.get('help.execution_thread_count'), type=int,
                                 default=config.get_int_value('execution.execution_thread_count', '4'),
                                 choices=facefusion.choices.execution_thread_count_range,
                                 metavar=create_metavar(facefusion.choices.execution_thread_count_range))
    group_execution.add_argument('--execution-queue-count', help=wording.get('help.execution_queue_count'), type=int,
                                 default=config.get_int_value('execution.execution_queue_count', '1'),
                                 choices=facefusion.choices.execution_queue_count_range,
                                 metavar=create_metavar(facefusion.choices.execution_queue_count_range))
    # memory
    group_memory = program.add_argument_group('memory')
    group_memory.add_argument('--video-memory-strategy', help=wording.get('help.video_memory_strategy'),
                              default=config.get_str_value('memory.video_memory_strategy', 'strict'),
                              choices=facefusion.choices.video_memory_strategies)
    group_memory.add_argument('--system-memory-limit', help=wording.get('help.system_memory_limit'), type=int,
                              default=config.get_int_value('memory.system_memory_limit', '0'),
                              choices=facefusion.choices.system_memory_limit_range,
                              metavar=create_metavar(facefusion.choices.system_memory_limit_range))
    # face analyser
    group_face_analyser = program.add_argument_group('face analyser')
    group_face_analyser.add_argument('--face-analyser-order', help=wording.get('help.face_analyser_order'),
                                     default=config.get_str_value('face_analyser.face_analyser_order', 'left-right'),
                                     choices=facefusion.choices.face_analyser_orders)
    group_face_analyser.add_argument('--face-analyser-age', help=wording.get('help.face_analyser_age'),
                                     default=config.get_str_value('face_analyser.face_analyser_age'),
                                     choices=facefusion.choices.face_analyser_ages)
    group_face_analyser.add_argument('--face-analyser-gender', help=wording.get('help.face_analyser_gender'),
                                     default=config.get_str_value('face_analyser.face_analyser_gender'),
                                     choices=facefusion.choices.face_analyser_genders)
    group_face_analyser.add_argument('--face-detector-model', help=wording.get('help.face_detector_model'),
                                     default=config.get_str_value('face_analyser.face_detector_model', 'yoloface'),
                                     choices=facefusion.choices.face_detector_set.keys())
    group_face_analyser.add_argument('--face-detector-size', help=wording.get('help.face_detector_size'),
                                     default=config.get_str_value('face_analyser.face_detector_size', '640x640'))
    group_face_analyser.add_argument('--face-detector-score', help=wording.get('help.face_detector_score'), type=float,
                                     default=config.get_float_value('face_analyser.face_detector_score', '0.5'),
                                     choices=facefusion.choices.face_detector_score_range,
                                     metavar=create_metavar(facefusion.choices.face_detector_score_range))
    group_face_analyser.add_argument('--face-landmarker-score', help=wording.get('help.face_landmarker_score'),
                                     type=float,
                                     default=config.get_float_value('face_analyser.face_landmarker_score', '0.5'),
                                     choices=facefusion.choices.face_landmarker_score_range,
                                     metavar=create_metavar(facefusion.choices.face_landmarker_score_range))
    # face selector
    group_face_selector = program.add_argument_group('face selector')
    group_face_selector.add_argument('--face-selector-mode', help=wording.get('help.face_selector_mode'),
                                     default=config.get_str_value('face_selector.face_selector_mode', 'reference'),
                                     choices=facefusion.choices.face_selector_modes)
    group_face_selector.add_argument('--reference-face-position', help=wording.get('help.reference_face_position'),
                                     type=int,
                                     default=config.get_int_value('face_selector.reference_face_position', '0'))
    group_face_selector.add_argument('--reference-face-distance', help=wording.get('help.reference_face_distance'),
                                     type=float,
                                     default=config.get_float_value('face_selector.reference_face_distance', '0.6'),
                                     choices=facefusion.choices.reference_face_distance_range,
                                     metavar=create_metavar(facefusion.choices.reference_face_distance_range))
    group_face_selector.add_argument('--reference-frame-number', help=wording.get('help.reference_frame_number'),
                                     type=int,
                                     default=config.get_int_value('face_selector.reference_frame_number', '0'))
    # face mask
    group_face_mask = program.add_argument_group('face mask')
    group_face_mask.add_argument('--face-mask-types', help=wording.get('help.face_mask_types').format(
        choices=', '.join(facefusion.choices.face_mask_types)),
                                 default=config.get_str_list('face_mask.face_mask_types', 'box'),
                                 choices=facefusion.choices.face_mask_types, nargs='+', metavar='FACE_MASK_TYPES')
    group_face_mask.add_argument('--face-mask-blur', help=wording.get('help.face_mask_blur'), type=float,
                                 default=config.get_float_value('face_mask.face_mask_blur', '0.3'),
                                 choices=facefusion.choices.face_mask_blur_range,
                                 metavar=create_metavar(facefusion.choices.face_mask_blur_range))
    group_face_mask.add_argument('--face-mask-padding', help=wording.get('help.face_mask_padding'), type=int,
                                 default=config.get_int_list('face_mask.face_mask_padding', '0 0 0 0'), nargs='+')
    group_face_mask.add_argument('--face-mask-regions', help=wording.get('help.face_mask_regions').format(
        choices=', '.join(facefusion.choices.face_mask_regions)),
                                 default=config.get_str_list('face_mask.face_mask_regions',
                                                             ' '.join(facefusion.choices.face_mask_regions)),
                                 choices=facefusion.choices.face_mask_regions, nargs='+', metavar='FACE_MASK_REGIONS')
    # frame extraction
    group_frame_extraction = program.add_argument_group('frame extraction')
    group_frame_extraction.add_argument('--trim-frame-start', help=wording.get('help.trim_frame_start'), type=int,
                                        default=facefusion.config.get_int_value('frame_extraction.trim_frame_start'))
    group_frame_extraction.add_argument('--trim-frame-end', help=wording.get('help.trim_frame_end'), type=int,
                                        default=facefusion.config.get_int_value('frame_extraction.trim_frame_end'))
    group_frame_extraction.add_argument('--temp-frame-format', help=wording.get('help.temp_frame_format'),
                                        default=config.get_str_value('frame_extraction.temp_frame_format', 'jpg'),
                                        choices=facefusion.choices.temp_frame_formats)
    group_frame_extraction.add_argument('--temp-frame-quality', help=wording.get('help.temp_frame_quality'), type=int,
                                        default=config.get_int_value('frame_extraction.temp_frame_quality', '100'),
                                        choices=facefusion.choices.temp_frame_quality_range,
                                        metavar=create_metavar(facefusion.choices.temp_frame_quality_range))
    group_frame_extraction.add_argument('--keep-temp', help=wording.get('help.keep_temp'), action='store_true',
                                        default=config.get_bool_value('frame_extraction.keep_temp'))
    # output creation
    group_output_creation = program.add_argument_group('output creation')
    group_output_creation.add_argument('--output-image-quality', help=wording.get('help.output_image_quality'),
                                       type=int,
                                       default=config.get_int_value('output_creation.output_image_quality', '80'),
                                       choices=facefusion.choices.output_image_quality_range,
                                       metavar=create_metavar(facefusion.choices.output_image_quality_range))
    group_output_creation.add_argument('--output-image-resolution', help=wording.get('help.output_image_resolution'),
                                       default=config.get_str_value('output_creation.output_image_resolution'))
    group_output_creation.add_argument('--output-video-encoder', help=wording.get('help.output_video_encoder'),
                                       default=config.get_str_value('output_creation.output_video_encoder', 'libx264'),
                                       choices=facefusion.choices.output_video_encoders)
    group_output_creation.add_argument('--output-video-preset', help=wording.get('help.output_video_preset'),
                                       default=config.get_str_value('output_creation.output_video_preset', 'veryfast'),
                                       choices=facefusion.choices.output_video_presets)
    group_output_creation.add_argument('--output-video-quality', help=wording.get('help.output_video_quality'),
                                       type=int,
                                       default=config.get_int_value('output_creation.output_video_quality', '80'),
                                       choices=facefusion.choices.output_video_quality_range,
                                       metavar=create_metavar(facefusion.choices.output_video_quality_range))
    group_output_creation.add_argument('--output-video-resolution', help=wording.get('help.output_video_resolution'),
                                       default=config.get_str_value('output_creation.output_video_resolution'))
    group_output_creation.add_argument('--output-video-fps', help=wording.get('help.output_video_fps'), type=float)
    group_output_creation.add_argument('--skip-audio', help=wording.get('help.skip_audio'), action='store_true',
                                       default=config.get_bool_value('output_creation.skip_audio'))
    # frame processors
    available_frame_processors = list_directory('facefusion/processors/frame/modules')
    program = ArgumentParser(parents=[program], formatter_class=program.formatter_class, add_help=True)
    group_frame_processors = program.add_argument_group('frame processors')
    group_frame_processors.add_argument('--frame-processors', help=wording.get('help.frame_processors').format(
        choices=', '.join(available_frame_processors)), default=config.get_str_list('frame_processors.frame_processors',
                                                                                    'face_swapper'), nargs='+')
    for frame_processor in available_frame_processors:
        frame_processor_module = load_frame_processor_module(frame_processor)
        frame_processor_module.register_args(group_frame_processors)
    # uis
    available_ui_layouts = list_directory('facefusion/uis/layouts')
    group_uis = program.add_argument_group('uis')
    group_uis.add_argument('--ui-layouts',
                           help=wording.get('help.ui_layouts').format(choices=', '.join(available_ui_layouts)),
                           default=config.get_str_list('uis.ui_layouts', 'default'), nargs='+')
    run(program)


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
    # memory
    facefusion.globals.video_memory_strategy = args.video_memory_strategy
    facefusion.globals.system_memory_limit = args.system_memory_limit
    # face analyser
    facefusion.globals.face_analyser_order = args.face_analyser_order
    facefusion.globals.face_analyser_age = args.face_analyser_age
    facefusion.globals.face_analyser_gender = args.face_analyser_gender
    facefusion.globals.face_detector_model = args.face_detector_model
    if args.face_detector_size in facefusion.choices.face_detector_set[args.face_detector_model]:
        facefusion.globals.face_detector_size = args.face_detector_size
    else:
        facefusion.globals.face_detector_size = '640x640'
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
    if is_image(args.target_path):
        output_image_resolution = detect_image_resolution(args.target_path)
        output_image_resolutions = create_image_resolutions(output_image_resolution)
        if args.output_image_resolution in output_image_resolutions:
            facefusion.globals.output_image_resolution = args.output_image_resolution
        else:
            facefusion.globals.output_image_resolution = pack_resolution(output_image_resolution)
    facefusion.globals.output_video_encoder = args.output_video_encoder
    facefusion.globals.output_video_preset = args.output_video_preset
    facefusion.globals.output_video_quality = args.output_video_quality
    if is_video(args.target_path):
        output_video_resolution = detect_video_resolution(args.target_path)
        output_video_resolutions = create_video_resolutions(output_video_resolution)
        if args.output_video_resolution in output_video_resolutions:
            facefusion.globals.output_video_resolution = args.output_video_resolution
        else:
            facefusion.globals.output_video_resolution = pack_resolution(output_video_resolution)
    if args.output_video_fps or is_video(args.target_path):
        facefusion.globals.output_video_fps = normalize_fps(args.output_video_fps) or detect_video_fps(args.target_path)
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
    if facefusion.globals.system_memory_limit > 0:
        limit_system_memory(facefusion.globals.system_memory_limit)
    if not pre_check() or not content_analyser.pre_check() or not face_analyser.pre_check() or not face_masker.pre_check():
        return
    for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
        if not frame_processor_module.pre_check():
            return
    if facefusion.globals.headless:
        conditional_process(None)
    else:
        import facefusion.uis.core as ui

        for ui_layout in ui.get_ui_layouts_modules(facefusion.globals.ui_layouts):
            if not ui_layout.pre_check():
                return
        ui.launch()


# def destroy() -> None:
#     process_manager.stop()
#     while process_manager.is_processing():
#         sleep(0.5)
#     if facefusion.globals.target_path:
#         clear_temp(facefusion.globals.target_path)
#     sys.exit(0)


def pre_check() -> bool:
    status = FFStatus()
    if sys.version_info < (3, 9):
        status.update(wording.get('python_not_supported').format(version='3.9'))
        return False
    if not shutil.which('ffmpeg'):
        status.update(wording.get('ffmpeg_not_installed'))
        return False
    return True


def conditional_process(job: JobParams) -> None:
    start_time = time.time()
    for frame_processor_module in get_frame_processors_modules(job.frame_processors):
        while not frame_processor_module.post_check():
            logger.disable()
            sleep(0.5)
        logger.enable()
        if not frame_processor_module.pre_process('output'):
            return
    conditional_append_reference_faces(job)
    target_path = job.target_path
    print(f"Processing {target_path}")
    try:
        if is_image(target_path):
            process_image(start_time, job)
        if is_video(target_path):
            reference_faces = job.reference_face_dict
            if len(reference_faces) > 1:
                average_face = None
                all_faces = []
                embedding_list = []
                normed_embedding_list = []
                first_key = None
                for key, faces in reference_faces.items():
                    if not first_key:
                        first_key = key
                    for face in faces:
                        all_faces.append(face)
                        embedding_list.append(face.embedding)
                        normed_embedding_list.append(face.normed_embedding)
                first_face = all_faces[0]
                average_face = Face(
                    bounding_box=first_face.bounding_box,
                    landmarks=first_face.landmarks,
                    scores=first_face.scores,
                    embedding=numpy.mean(embedding_list, axis=0),
                    normed_embedding=numpy.mean(normed_embedding_list, axis=0),
                    gender=first_face.gender,
                    age=first_face.age
                )
                reference_faces = {first_key: [average_face]}
                job.reference_face_dict = reference_faces
            process_video(start_time, job)
    except Exception as e:
        print(f"Exception Processing: {e} at {traceback.print_exc()}")
        traceback.print_exc()


def conditional_append_reference_faces(job=None) -> None:
    if not job:
        job = JobParams().from_globals()
    if 'reference' in job.face_selector_mode and not get_reference_faces():
        source_frames = read_static_images(job.source_paths)
        source_face = get_average_face(source_frames)
        source_frames_2 = read_static_images(job.source_paths_2)
        source_face_2 = get_average_face(source_frames_2)
        if is_video(job.target_path):
            reference_frame = get_video_frame(job.target_path, job.reference_frame_number)
        else:
            reference_frame = read_image(job.target_path)
        reference_face = get_one_face(reference_frame, job.reference_face_position)
        append_reference_face('origin', reference_face)
        append_reference_face('origin', reference_face, True)
        if reference_face and (source_face or source_face_2):
            for frame_processor_module in get_frame_processors_modules(job.frame_processors):
                for src_face, is_second in [(source_face, False), (source_face_2, True)]:  # source_face, source_face_2
                    if src_face:
                        abstract_reference_frame = frame_processor_module.get_reference_frame(src_face, reference_face,
                                                                                              reference_frame)
                        if numpy.any(abstract_reference_frame):
                            reference_frame = abstract_reference_frame
                            reference_face = get_one_face(reference_frame, job.reference_face_position)
                            append_reference_face(frame_processor_module.__name__, reference_face, is_second)


def process_image(start_time: float, job: JobParams) -> None:
    status = FFStatus()
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
        frame_processor_module.process_image(job.source_paths, job.source_paths_2, job.output_path, job.output_path)
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
        seconds = '{:.2f}'.format((time.time() - start_time) % 60)
        status_str = "Processing completed in " + seconds + " seconds."
    else:
        status_str = wording.get('processing_image_failed')
    status.update(status_str)


def process_video(start_time, job) -> None:
    print(f"Processing video: {job.target_path}")
    status = FFStatus()
    if analyse_video(job.target_path, job.trim_frame_start,
                     job.trim_frame_end):
        status.update("Naughty naughty!!")
        status.cancel()
        return
    status.update("Processing facefusion video.")
    fps = detect_video_fps(job.target_path) if job.keep_fps else 25.0
    # create temp
    create_temp(job.target_path)

    # extract frames
    status.update(f"Extracting frames from {os.path.basename(job.target_path)}...")
    extract_frames(job.target_path, job.output_video_resolution, fps)
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
            frame_processor_module.process_video(job.source_paths, job.source_paths_2, temp_frame_paths)
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
    if not merge_video(job.target_path, fps):
        status.update(wording.get('merging_video_failed'))
    # handle audio
    failed = False
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
            if 'lip_syncer' in facefusion.globals.frame_processors:
                print("Running lip syncer...")
                source_audio_path = get_first(filter_audio_paths(facefusion.globals.source_paths))
                if source_audio_path and replace_audio(facefusion.globals.target_path, source_audio_path,
                                                       facefusion.globals.output_path):
                    logger.info(wording.get('restoring_audio_succeed'), __name__.upper())
                else:
                    logger.warn(wording.get('restoring_audio_skipped'), __name__.upper())
                    move_temp(facefusion.globals.target_path, facefusion.globals.output_path)
            else:
                print("Running normal audio restore...")
                if restore_audio(facefusion.globals.target_path, facefusion.globals.output_path,
                                 facefusion.globals.output_video_fps):
                    logger.info(wording.get('restoring_audio_succeed'), __name__.upper())
                else:
                    logger.warn(wording.get('restoring_audio_skipped'), __name__.upper())
                    move_temp(facefusion.globals.target_path, facefusion.globals.output_path)

        except Exception as f:
            print(f"Failed to restore audio: {f} at {traceback.print_exc()}")
            failed = True
    # clear temp
    if not failed:
        status.update(wording.get('clearing_temp'))
        clear_temp()
    # validate video
    if is_video(job.target_path):
        print("Processing completed.")
        status.update(wording.get('processing_video_succeed'))
    else:
        print("Processing failed.")
        status.update(wording.get('processing_video_failed'))
