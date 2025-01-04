import shutil
import sys
from time import time

import numpy

from facefusion import logger, process_manager, state_manager, wording
from facefusion.args import apply_args, collect_job_args, reduce_step_args
from facefusion.common_helper import get_first
from facefusion.exit_helper import conditional_exit, hard_exit
from facefusion.face_store import clear_reference_faces
from facefusion.ffmpeg import copy_image, extract_frames, finalize_image, merge_video, replace_audio, restore_audio
from facefusion.filesystem import filter_audio_paths, is_image, is_video
from facefusion.jobs import job_helper, job_manager, job_runner
from facefusion.jobs.job_list import compose_job_list
from facefusion.memory import limit_system_memory
from facefusion.processors.core import get_processors_modules
from facefusion.statistics import conditional_log_statistics
from facefusion.temp_helper import clear_temp_directory, create_temp_directory, get_temp_file_path, \
    get_temp_frame_paths, move_temp_file
from facefusion.typing import Args, ErrorCode, Face
from facefusion.vision import pack_resolution, restrict_image_resolution, \
    restrict_video_fps, restrict_video_resolution, unpack_resolution
from facefusion.workers.classes.content_analyser import ContentAnalyser
from facefusion.workers.core import get_worker_modules


def route(args: Args) -> None:
    system_memory_limit = state_manager.get_item('system_memory_limit')
    if system_memory_limit and system_memory_limit > 0:
        limit_system_memory(system_memory_limit)
    if state_manager.get_item('command') == 'force-download':
        error_code = force_download()
        return conditional_exit(error_code)
    if state_manager.get_item('command') in ['job-list', 'job-create', 'job-submit', 'job-submit-all', 'job-delete',
                                             'job-delete-all', 'job-add-step', 'job-remix-step', 'job-insert-step',
                                             'job-remove-step']:
        if not job_manager.init_jobs(state_manager.get_item('jobs_path')):
            hard_exit(1)
        error_code = route_job_manager(args)
        hard_exit(error_code)
    if not pre_check():
        return conditional_exit(2)
    if state_manager.get_item('command') == 'run':
        import facefusion.uis.core as ui

        if not common_pre_check() or not processors_pre_check():
            return conditional_exit(2)
        for ui_layout in ui.get_ui_layouts_modules(state_manager.get_item('ui_layouts')):
            if not ui_layout.pre_check():
                return conditional_exit(2)
        ui.launch()
    if state_manager.get_item('command') == 'headless-run':
        if not job_manager.init_jobs(state_manager.get_item('jobs_path')):
            hard_exit(1)
        error_core = process_headless(args)
        hard_exit(error_core)
    if state_manager.get_item('command') in ['job-run', 'job-run-all', 'job-retry', 'job-retry-all']:
        if not job_manager.init_jobs(state_manager.get_item('jobs_path')):
            hard_exit(1)
        error_code = route_job_runner()
        hard_exit(error_code)


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        logger.error(wording.get('python_not_supported').format(version='3.9'), __name__)
        return False
    if not shutil.which('curl'):
        logger.error(wording.get('curl_not_installed'), __name__)
        return False
    if not shutil.which('ffmpeg'):
        logger.error(wording.get('ffmpeg_not_installed'), __name__)
        return False
    return True


def common_pre_check() -> bool:
    modules = get_worker_modules()
    return all(module.pre_check() for module in modules)


def processors_pre_check() -> bool:
    for processor_module in get_processors_modules(state_manager.get_item('processors')):
        if not processor_module.pre_check():
            return False
    return True


def conditional_process() -> ErrorCode:
    start_time = time()
    for processor_module in get_processors_modules(state_manager.get_item('processors')):
        if not processor_module.pre_process('output'):
            return 2
    #average_reference_faces()
    if is_image(state_manager.get_item('target_path')):
        return process_image(start_time)
    if is_video(state_manager.get_item('target_path')):
        return process_video(start_time)
    return 0


def force_download() -> ErrorCode:
    all_workers = get_worker_modules()
    all_processors = get_processors_modules()
    for module in all_workers + all_processors:
        if not module.download_all_models():
            return 2
    return 0


def route_job_manager(args: Args) -> ErrorCode:
    if state_manager.get_item('command') == 'job-list':
        job_headers, job_contents = compose_job_list(state_manager.get_item('job_status'))

        if job_contents:
            logger.table(job_headers, job_contents)
            return 0
        return 1
    if state_manager.get_item('command') == 'job-create':
        if job_manager.create_job(state_manager.get_item('job_id')):
            logger.info(wording.get('job_created').format(job_id=state_manager.get_item('job_id')), __name__)
            return 0
        logger.error(wording.get('job_not_created').format(job_id=state_manager.get_item('job_id')), __name__)
        return 1
    if state_manager.get_item('command') == 'job-submit':
        if job_manager.submit_job(state_manager.get_item('job_id')):
            logger.info(wording.get('job_submitted').format(job_id=state_manager.get_item('job_id')), __name__)
            return 0
        logger.error(wording.get('job_not_submitted').format(job_id=state_manager.get_item('job_id')), __name__)
        return 1
    if state_manager.get_item('command') == 'job-submit-all':
        if job_manager.submit_jobs():
            logger.info(wording.get('job_all_submitted'), __name__)
            return 0
        logger.error(wording.get('job_all_not_submitted'), __name__)
        return 1
    if state_manager.get_item('command') == 'job-delete':
        if job_manager.delete_job(state_manager.get_item('job_id')):
            logger.info(wording.get('job_deleted').format(job_id=state_manager.get_item('job_id')), __name__)
            return 0
        logger.error(wording.get('job_not_deleted').format(job_id=state_manager.get_item('job_id')), __name__)
        return 1
    if state_manager.get_item('command') == 'job-delete-all':
        if job_manager.delete_jobs():
            logger.info(wording.get('job_all_deleted'), __name__)
            return 0
        logger.error(wording.get('job_all_not_deleted'), __name__)
        return 1
    if state_manager.get_item('command') == 'job-add-step':
        step_args = reduce_step_args(args)

        if job_manager.add_step(state_manager.get_item('job_id'), step_args):
            logger.info(wording.get('job_step_added').format(job_id=state_manager.get_item('job_id')), __name__)
            return 0
        logger.error(wording.get('job_step_not_added').format(job_id=state_manager.get_item('job_id')), __name__)
        return 1
    if state_manager.get_item('command') == 'job-remix-step':
        step_args = reduce_step_args(args)

        if job_manager.remix_step(state_manager.get_item('job_id'), state_manager.get_item('step_index'), step_args):
            logger.info(wording.get('job_remix_step_added').format(job_id=state_manager.get_item('job_id'),
                                                                   step_index=state_manager.get_item('step_index')),
                        __name__)
            return 0
        logger.error(wording.get('job_remix_step_not_added').format(job_id=state_manager.get_item('job_id'),
                                                                    step_index=state_manager.get_item('step_index')),
                     __name__)
        return 1
    if state_manager.get_item('command') == 'job-insert-step':
        step_args = reduce_step_args(args)

        if job_manager.insert_step(state_manager.get_item('job_id'), state_manager.get_item('step_index'), step_args):
            logger.info(wording.get('job_step_inserted').format(job_id=state_manager.get_item('job_id'),
                                                                step_index=state_manager.get_item('step_index')),
                        __name__)
            return 0
        logger.error(wording.get('job_step_not_inserted').format(job_id=state_manager.get_item('job_id'),
                                                                 step_index=state_manager.get_item('step_index')),
                     __name__)
        return 1
    if state_manager.get_item('command') == 'job-remove-step':
        if job_manager.remove_step(state_manager.get_item('job_id'), state_manager.get_item('step_index')):
            logger.info(wording.get('job_step_removed').format(job_id=state_manager.get_item('job_id'),
                                                               step_index=state_manager.get_item('step_index')),
                        __name__)
            return 0
        logger.error(wording.get('job_step_not_removed').format(job_id=state_manager.get_item('job_id'),
                                                                step_index=state_manager.get_item('step_index')),
                     __name__)
        return 1
    return 1


def route_job_runner() -> ErrorCode:
    if state_manager.get_item('command') == 'job-run':
        logger.info(wording.get('running_job').format(job_id=state_manager.get_item('job_id')), __name__)
        if job_runner.run_job(state_manager.get_item('job_id'), process_step):
            logger.info(wording.get('processing_job_succeed').format(job_id=state_manager.get_item('job_id')), __name__)
            return 0
        logger.info(wording.get('processing_job_failed').format(job_id=state_manager.get_item('job_id')), __name__)
        return 1
    if state_manager.get_item('command') == 'job-run-all':
        logger.info(wording.get('running_jobs'), __name__)
        if job_runner.run_jobs(process_step):
            logger.info(wording.get('processing_jobs_succeed'), __name__)
            return 0
        logger.info(wording.get('processing_jobs_failed'), __name__)
        return 1
    if state_manager.get_item('command') == 'job-retry':
        logger.info(wording.get('retrying_job').format(job_id=state_manager.get_item('job_id')), __name__)
        if job_runner.retry_job(state_manager.get_item('job_id'), process_step):
            logger.info(wording.get('processing_job_succeed').format(job_id=state_manager.get_item('job_id')), __name__)
            return 0
        logger.info(wording.get('processing_job_failed').format(job_id=state_manager.get_item('job_id')), __name__)
        return 1
    if state_manager.get_item('command') == 'job-retry-all':
        logger.info(wording.get('retrying_jobs'), __name__)
        if job_runner.retry_jobs(process_step):
            logger.info(wording.get('processing_jobs_succeed'), __name__)
            return 0
        logger.info(wording.get('processing_jobs_failed'), __name__)
        return 1
    return 2


def process_step(job_id: str, step_index: int, step_args: Args) -> bool:
    clear_reference_faces()
    step_total = job_manager.count_step_total(job_id)
    step_args.update(collect_job_args())
    apply_args(step_args, True)

    logger.info(wording.get('processing_step').format(step_current=step_index + 1, step_total=step_total), __name__)
    if common_pre_check() and processors_pre_check():
        error_code = conditional_process()
        return error_code == 0
    return False


def process_headless(args: Args) -> ErrorCode:
    job_id = job_helper.suggest_job_id('headless')
    step_args = reduce_step_args(args)

    if job_manager.create_job(job_id) and job_manager.add_step(job_id, step_args) and job_manager.submit_job(
            job_id) and job_runner.run_job(job_id, process_step):
        return 0
    return 1


def process_image(start_time: float) -> ErrorCode:
    analyser = ContentAnalyser()
    if analyser.analyse_image(state_manager.get_item('target_path')):
        return 3
    # clear temp
    logger.debug(wording.get('clearing_temp'), __name__)
    clear_temp_directory(state_manager.get_item('target_path'))
    # create temp
    logger.debug(wording.get('creating_temp'), __name__)
    create_temp_directory(state_manager.get_item('target_path'))
    # copy image
    process_manager.start()
    temp_image_resolution = pack_resolution(restrict_image_resolution(state_manager.get_item('target_path'),
                                                                      unpack_resolution(state_manager.get_item(
                                                                          'output_image_resolution'))))
    logger.info(wording.get('copying_image').format(resolution=temp_image_resolution), __name__)
    if copy_image(state_manager.get_item('target_path'), temp_image_resolution):
        logger.debug(wording.get('copying_image_succeed'), __name__)
    else:
        logger.error(wording.get('copying_image_failed'), __name__)
        process_manager.end()
        return 1
    # process image
    temp_file_path = get_temp_file_path(state_manager.get_item('target_path'))
    for processor_module in get_processors_modules(state_manager.get_item('processors')):
        logger.info(wording.get('processing'), processor_module.display_name)
        processor_module.process_image(temp_file_path, temp_file_path)
        processor_module.post_process()
    if is_process_stopping():
        process_manager.end()
        return 4
    # finalize image
    logger.info(wording.get('finalizing_image').format(resolution=state_manager.get_item('output_image_resolution')),
                __name__)
    if finalize_image(state_manager.get_item('target_path'), state_manager.get_item('output_path'),
                      state_manager.get_item('output_image_resolution')):
        logger.debug(wording.get('finalizing_image_succeed'), __name__)
    else:
        logger.warn(wording.get('finalizing_image_skipped'), __name__)
    # clear temp
    logger.debug(wording.get('clearing_temp'), __name__)
    clear_temp_directory(state_manager.get_item('target_path'))
    # validate image
    if is_image(state_manager.get_item('output_path')):
        seconds = '{:.2f}'.format((time() - start_time) % 60)
        logger.info(wording.get('processing_image_succeed').format(seconds=seconds), __name__)
        conditional_log_statistics()
    else:
        logger.error(wording.get('processing_image_failed'), __name__)
        process_manager.end()
        return 1
    process_manager.end()
    return 0


def process_video(start_time: float) -> ErrorCode:
    analyser = ContentAnalyser()
    if analyser.analyse_video(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'),
                     state_manager.get_item('trim_frame_end')):
        return 3
    # clear temp
    logger.debug(wording.get('clearing_temp'), __name__)
    clear_temp_directory(state_manager.get_item('target_path'))
    # create temp
    logger.debug(wording.get('creating_temp'), __name__)
    create_temp_directory(state_manager.get_item('target_path'))
    # extract frames
    print(f"Starting process manager")
    process_manager.start()
    temp_video_resolution = pack_resolution(restrict_video_resolution(state_manager.get_item('target_path'),
                                                                      unpack_resolution(state_manager.get_item(
                                                                          'output_video_resolution'))))
    temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'),
                                        state_manager.get_item('output_video_fps'))
    logger.info(wording.get('extracting_frames').format(resolution=temp_video_resolution, fps=temp_video_fps), __name__)
    if extract_frames(state_manager.get_item('target_path'), temp_video_resolution, temp_video_fps):
        logger.debug(wording.get('extracting_frames_succeed'), __name__)
    else:
        if is_process_stopping():
            process_manager.end()
            return 4
        logger.error(wording.get('extracting_frames_failed'), __name__)
        process_manager.end()
        return 1
    # process frames
    temp_frame_paths = get_temp_frame_paths(state_manager.get_item('target_path'))
    if temp_frame_paths:
        for processor_module in get_processors_modules(state_manager.get_item('processors')):
            print(f"Processing {processor_module.display_name}")
            logger.info(wording.get('processing'), processor_module.display_name)
            processor_module.process_video(temp_frame_paths)
            print(f"Post processing {processor_module.display_name}")
            processor_module.post_process()
            print(f"Post processing {processor_module.display_name} done in {time() - start_time} seconds")
        if is_process_stopping():
            return 4
    else:
        logger.error(wording.get('temp_frames_not_found'), __name__)
        process_manager.end()
        return 1
    # merge video
    logger.info(wording.get('merging_video').format(resolution=state_manager.get_item('output_video_resolution'),
                                                    fps=state_manager.get_item('output_video_fps')), __name__)
    print(f"Merging video")
    if merge_video(state_manager.get_item('target_path'), state_manager.get_item('output_video_resolution'),
                   state_manager.get_item('output_video_fps')):
        logger.debug(wording.get('merging_video_succeed'), __name__)
        print(f"Merging video succeed")
    else:
        if is_process_stopping():
            process_manager.end()
            return 4
        logger.error(wording.get('merging_video_failed'), __name__)
        print(f"Merging video failed")
        process_manager.end()
        return 1
    # handle audio
    if state_manager.get_item('skip_audio'):
        logger.info(wording.get('skipping_audio'), __name__)
        move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
    else:
        source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
        if source_audio_path:
            print(f"Replacing audio")
            if replace_audio(state_manager.get_item('target_path'), source_audio_path,
                             state_manager.get_item('output_path')):
                logger.debug(wording.get('replacing_audio_succeed'), __name__)
            else:
                if is_process_stopping():
                    process_manager.end()
                    return 4
                logger.warn(wording.get('replacing_audio_skipped'), __name__)
                move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
        else:
            print(f"Restoring audio")
            if restore_audio(state_manager.get_item('target_path'), state_manager.get_item('output_path'),
                             state_manager.get_item('output_video_fps')):
                logger.debug(wording.get('restoring_audio_succeed'), __name__)
                print(f"Restoring audio succeed")
            else:
                if is_process_stopping():
                    process_manager.end()
                    return 4
                logger.warn(wording.get('restoring_audio_skipped'), __name__)
                print(f"Restoring audio skipped")
                move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
    # clear temp
    logger.debug(wording.get('clearing_temp'), __name__)
    clear_temp_directory(state_manager.get_item('target_path'))
    # validate video
    if is_video(state_manager.get_item('output_path')):
        seconds = '{:.2f}'.format((time() - start_time))
        logger.info(wording.get('processing_video_succeed').format(seconds=seconds), __name__)
        conditional_log_statistics()
    else:
        logger.error(wording.get('processing_video_failed'), __name__)
        process_manager.end()
        return 1
    process_manager.end()
    return 0


def is_process_stopping() -> bool:
    if process_manager.is_stopping():
        process_manager.end()
        logger.info(wording.get('processing_stopped'), __name__)
    return process_manager.is_pending()
