from typing import Any, Dict, Optional

WORDING : Dict[str, Any] =\
{
    'python_not_supported': 'Python version is not supported, upgrade to {version} or higher',
    'ffmpeg_not_installed': 'FFMpeg is not installed',
    'creating_temp': 'Creating temporary resources',
    'extracting_frames_fps': 'Extracting frames with {video_fps} FPS',
    'analysing': 'Analysing',
    'processing': 'Processing',
    'downloading': 'Downloading',
    'temp_frames_not_found': 'Temporary frames not found',
    'compressing_image_succeed': 'Compressing image succeed',
    'compressing_image_skipped': 'Compressing image skipped',
    'merging_video_fps': 'Merging video with {video_fps} FPS',
    'merging_video_failed': 'Merging video failed',
    'skipping_audio': 'Skipping audio',
    'restoring_audio_succeed': 'Restoring audio succeed',
    'restoring_audio_skipped': 'Restoring audio skipped',
    'clearing_temp': 'Clearing temporary resources',
    'processing_image_succeed': 'Processing to image succeed in {seconds} seconds',
    'processing_image_failed': 'Processing to image failed',
    'processing_video_succeed': 'Processing to video succeed in {seconds} seconds',
    'processing_video_failed': 'Processing to video failed',
    'model_download_not_done': 'Download of the model is not done',
    'model_file_not_present': 'File of the model is not present',
    'select_image_source': 'Select a image for source path',
    'select_audio_source': 'Select a audio for source path',
    'select_video_target': 'Select a video for target path',
    'select_image_or_video_target': 'Select a image or video for target path',
    'select_file_or_directory_output': 'Select a file or directory for output path',
    'no_source_face_detected': 'No source face detected',
    'frame_processor_not_loaded': 'Frame processor {frame_processor} could not be loaded',
    'frame_processor_not_implemented': 'Frame processor {frame_processor} not implemented correctly',
    'ui_layout_not_loaded': 'UI layout {ui_layout} could not be loaded',
    'ui_layout_not_implemented': 'UI layout {ui_layout} not implemented correctly',
    'stream_not_loaded': 'Stream {stream_mode} could not be loaded',
    'point': '.',
    'comma': ',',
    'colon': ':',
    'question_mark': '?',
    'exclamation_mark': '!',
    'help':
    {
        # installer
        'install_dependency': 'select the variant of {dependency} to install',
        'skip_venv': 'skip the virtual environment check',
        # general
        'source': 'choose single or multiple source images or audios',
        'target': 'choose single target image or video',
        'output': 'specify the output file or directory',
        # misc
        'skip_download': 'omit automate downloads and remote lookups',
        'headless': 'run the program without a user interface',
        'log_level': 'adjust the message severity displayed in the terminal',
        # execution
        'execution_providers': 'accelerate the model inference using different providers (choices: {choices}, ...)',
        'execution_thread_count': 'specify the amount of parallel threads while processing',
        'execution_queue_count': 'specify the amount of frames each thread is processing',
        # memory
        'video_memory_strategy': 'balance fast frame processing and low vram usage',
        'system_memory_limit': 'limit the available ram that can be used while processing',
        # face analyser
        'face_analyser_order': 'specify the order in which the face analyser detects faces.',
        'face_analyser_age': 'filter the detected faces based on their age',
        'face_analyser_gender': 'filter the detected faces based on their gender',
        'face_detector_model': 'choose the model responsible for detecting the face',
        'face_detector_size': 'specify the size of the frame provided to the face detector',
        'face_detector_score': 'filter the detected faces base on the confidence score',
        # face selector
        'face_selector_mode': 'use reference based tracking with simple matching',
        'reference_face_position': 'specify the position used to create the reference face',
        'reference_face_distance': 'specify the desired similarity between the reference face and target face',
        'reference_frame_number': 'specify the frame used to create the reference face',
        # face mask
        'face_mask_types': 'mix and match different face mask types (choices: {choices})',
        'face_mask_blur': 'specify the degree of blur applied the box mask',
        'face_mask_padding': 'apply top, right, bottom and left padding to the box mask',
        'face_mask_regions': 'choose the facial features used for the region mask (choices: {choices})',
        # frame extraction
        'trim_frame_start': 'specify the the start frame of the target video',
        'trim_frame_end': 'specify the the end frame of the target video',
        'temp_frame_format': 'specify the temporary resources format',
        'temp_frame_quality': 'specify the temporary resources quality',
        'keep_temp': 'keep the temporary resources after processing',
        # output creation
        'output_image_quality': 'specify the image quality which translates to the compression factor',
        'output_video_encoder': 'specify the encoder use for the video compression',
        'output_video_preset': 'balance fast video processing and video file size',
        'output_video_quality': 'specify the video quality which translates to the compression factor',
        'output_video_resolution': 'specify the video output resolution based on the target video',
        'output_video_fps': 'specify the video output fps based on the target video',
        'skip_audio': 'omit the audio from the target video',
        # frame processors
        'frame_processors': 'load a single or multiple frame processors. (choices: {choices}, ...)',
        'face_debugger_items': 'load a single or multiple frame processors (choices: {choices})',
        'face_enhancer_model': 'choose the model responsible for enhancing the face',
        'face_enhancer_blend': 'blend the enhanced into the previous face',
        'face_swapper_model': 'choose the model responsible for swapping the face',
        'frame_enhancer_model': 'choose the model responsible for enhancing the frame',
        'frame_enhancer_blend': 'blend the enhanced into the previous frame',
        'lip_syncer_model': 'choose the model responsible for syncing the lips',
        # uis
        'ui_layouts': 'launch a single or multiple UI layouts (choices: {choices}, ...)'
    },
    'uis':
        {
            'start_button': 'Start',
            'stop_button': 'Stop',
            'clear_button': 'Clear',
            'donate_button': 'Donate',
            'benchmark_results_dataframe': 'Benchmark Results',
            'benchmark_runs_checkbox_group': 'Benchmark Runs',
            'benchmark_cycles_slider': 'Benchmark Cycles',
            'common_options_checkbox_group': 'Options',
            'execution_providers_checkbox_group': 'Execution Providers',
            'execution_queue_count_slider': 'Execution Queue Count',
            'execution_thread_count_slider': 'Execution Thread Count',
            'face_analyser_order_dropdown': 'Face Analyser Order',
            'face_analyser_age_dropdown': 'Face Analyser Age',
            'face_analyser_gender_dropdown': 'Face Analyser Gender',
            'face_detector_model_dropdown': 'Face Detector Model',
            'face_detector_size_dropdown': 'Face Detector Size',
            'face_detector_score_slider': 'Face Detector Score',
            'face_mask_types_checkbox_group': 'Face Mask Types',
            'face_mask_blur_slider': 'Face Mask Blur',
            'face_mask_padding_top_slider': 'Face Mask Padding Top',
            'face_mask_padding_right_slider': 'Face Mask Padding Right',
            'face_mask_padding_bottom_slider': 'Face Mask Padding Bottom',
            'face_mask_padding_left_slider': 'Face Mask Padding Left',
            'face_mask_region_checkbox_group': 'Face Mask Regions',
            'face_selector_mode_dropdown': 'Face Selector Mode',
            'reference_face_gallery': 'Reference Face',
            'reference_face_distance_slider': 'Reference Face Distance',
            'frame_processors_checkbox_group': 'Frame Processors',
            'face_debugger_items_checkbox_group': 'Face Debugger Items',
            'face_enhancer_model_dropdown': 'Face Enhancer Model',
            'face_enhancer_blend_slider': 'Face Enhancer Blend',
            'face_swapper_model_dropdown': 'Face Swapper Model',
            'frame_enhancer_model_dropdown': 'Frame Enhancer Model',
            'frame_enhancer_blend_slider': 'Frame Enhancer Blend',
            'video_memory_strategy_dropdown': 'Video Memory Strategy',
            'system_memory_limit_slider': 'System Memory Limit',
            'output_image_or_video': 'Output',
            'output_path_textbox': 'Output Path',
            'output_image_quality_slider': 'Output Image Quality',
            'output_video_encoder_dropdown': 'Output Video Encoder',
            'output_video_preset_dropdown': 'Output Video Preset',
            'output_video_quality_slider': 'Output Video Quality',
            'output_video_resolution_dropdown': 'Output Video Resolution',
            'output_video_fps_slider': 'Output Video Fps',
            'preview_image': 'Preview',
            'preview_frame_slider': 'Preview Frame',
            'source_file': 'Source',
            'target_file': 'Target',
            'temp_frame_format_dropdown': 'Temp Frame Format',
            'temp_frame_quality_slider': 'Temp Frame Quality',
            'trim_frame_start_slider': 'Trim Frame Start',
            'trim_frame_end_slider': 'Trim Frame End',
            'webcam_image': 'Webcam',
            'webcam_mode_radio': 'Webcam Mode',
            'webcam_resolution_dropdown': 'Webcam Resolution',
            'webcam_fps_slider': 'Webcam Fps'
        }

}


def get(key : str) -> Optional[str]:
    if '.' in key:
        section, name = key.split('.')
        if section in WORDING and name in WORDING[section]:
            return WORDING[section][name]
    if key in WORDING:
        return WORDING[key]
    return None
