from facefusion import state_manager
from facefusion.filesystem import is_image, is_video, list_directory
from facefusion.jobs import job_store
from facefusion.normalizer import normalize_fps, normalize_padding
from facefusion.processors.core import get_processors_modules
from facefusion.typing import Args
from facefusion.vision import create_image_resolutions, create_video_resolutions, detect_image_resolution, \
    detect_video_fps, detect_video_resolution, pack_resolution


def reduce_step_args(args: Args) -> Args:
    step_args = \
        {
            key: args[key] for key in args if key in job_store.get_step_keys()
        }
    return step_args


def collect_step_args() -> Args:
    step_args = \
        {
            key: state_manager.get_item(key) for key in job_store.get_step_keys()  # type:ignore[arg-type]
        }
    return step_args


def collect_job_args() -> Args:
    job_args = \
        {
            key: state_manager.get_item(key) for key in job_store.get_job_keys()  # type:ignore[arg-type]
        }
    return job_args


def apply_args(args: Args, apply_state_item: bool) -> None:
    if apply_state_item:
        cmd = state_manager.set_item
    else:
        cmd = state_manager.init_item

    # These keys require special handling, so we'll manage them first
    special_keys = {
        'face_mask_padding',
        'output_image_resolution',
        'output_video_resolution',
        'output_video_fps',
        'target_path',  # used for is_image / is_video checks
    }

    # 1) face_mask_padding -> normalize
    if 'face_mask_padding' in args:
        cmd('face_mask_padding', normalize_padding(args.get('face_mask_padding')))

    # 2) image resolution logic
    if is_image(args.get('target_path')):
        output_image_resolution = detect_image_resolution(args.get('target_path'))
        output_image_resolutions = create_image_resolutions(output_image_resolution)
        if args.get('output_image_resolution') in output_image_resolutions:
            cmd('output_image_resolution', args.get('output_image_resolution'))
        else:
            cmd('output_image_resolution', pack_resolution(output_image_resolution))

    # 3) video resolution logic
    if is_video(args.get('target_path')):
        output_video_resolution = detect_video_resolution(args.get('target_path'))
        output_video_resolutions = create_video_resolutions(output_video_resolution)
        if args.get('output_video_resolution') in output_video_resolutions:
            cmd('output_video_resolution', args.get('output_video_resolution'))
        else:
            cmd('output_video_resolution', pack_resolution(output_video_resolution))

    # 4) video fps logic
    if args.get('output_video_fps') is not None or is_video(args.get('target_path')):
        output_video_fps = normalize_fps(args.get('output_video_fps')) or detect_video_fps(args.get('target_path'))
        cmd('output_video_fps', output_video_fps)

    # For all other keys, just apply them directly
    for key, value in args.items():
        # We still want to store 'target_path' if present
        if key in special_keys:
            if key == 'target_path':
                cmd(key, value)
            continue
        cmd(key, value)

    # Let each processor module apply its own arguments
    for processor_module in get_processors_modules():
        try:
            processor_module.apply_args(args, cmd)
        except Exception as e:
            print(f"Error while applying args for processor {processor_module}: {e}")



def apply_globals(globals_dict: dict, init: bool = True) -> None:
    keys = [
        # general
        'command',
        # paths
        'jobs_path', 'source_paths', 'target_path', 'output_path',
        # face detector
        'face_detector_model', 'face_detector_size', 'face_detector_angles', 'face_detector_score',
        # face landmarker
        'face_landmarker_model', 'face_landmarker_score',
        # face selector
        'face_selector_mode', 'face_selector_order', 'face_selector_age_start',
        'face_selector_age_end', 'face_selector_gender', 'face_selector_race',
        'reference_face_position', 'reference_face_distance', 'reference_frame_number',
        # face masker
        'face_mask_types', 'face_mask_blur', 'face_mask_padding', 'face_mask_regions',
        # frame extraction
        'trim_frame_start', 'trim_frame_end', 'temp_frame_format', 'keep_temp',
        # output creation
        'output_image_quality', 'output_image_resolution', 'output_audio_encoder',
        'output_video_encoder', 'output_video_preset', 'output_video_quality',
        'output_video_resolution', 'output_video_fps', 'skip_audio',
        # processors
        'processors',
        # uis
        'open_browser', 'ui_layouts', 'ui_workflow',
        # execution
        'execution_device_id', 'execution_providers', 'execution_thread_count', 'execution_queue_count',
        # memory
        'video_memory_strategy', 'system_memory_limit',
        # misc
        'skip_download', 'log_level',
        # jobs
        'job_id', 'job_status', 'step_index'
    ]

    for key in keys:
        if key in globals_dict:
            value = globals_dict[key]
            if key == 'face_mask_padding':
                value = normalize_padding(value)
            elif key == 'output_video_fps':
                value = normalize_fps(value)
            elif key == 'output_image_resolution' and 'target_path' in globals_dict and is_image(
                    globals_dict['target_path']):
                output_image_resolution = detect_image_resolution(globals_dict['target_path'])
                output_image_resolutions = create_image_resolutions(output_image_resolution)
                if value not in output_image_resolutions:
                    value = pack_resolution(output_image_resolution)
            elif key == 'output_video_resolution' and 'target_path' in globals_dict and is_video(
                    globals_dict['target_path']):
                output_video_resolution = detect_video_resolution(globals_dict['target_path'])
                output_video_resolutions = create_video_resolutions(output_video_resolution)
                if value not in output_video_resolutions:
                    value = pack_resolution(output_video_resolution)
            elif key == 'output_video_fps' or (key == 'output_video_fps' and is_video(globals_dict['target_path'])):
                value = value or detect_video_fps(globals_dict['target_path'])

            if init:
                state_manager.init_item(key, value)
            else:
                state_manager.set_item(key, value)
        else:
            print(f"Key {key} not found in globals_dict.")

