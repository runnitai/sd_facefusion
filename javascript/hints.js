let hints_set = false;

const hint_dict = {
    "ff3_instant_runner_start_button": "Start processing the current job immediately.",
    "ff3_instant_runner_stop_button": "Stop processing the current job.",
    "ff3_instant_runner_clear_button": "Remove the current job from the list.",
    "ff3_processors_checkbox_group": "Select frame processors to apply during rendering.",
    "ff3_face_swapper_model_dropdown": "Choose the model used for face swapping.",
    "ff3_face_enhancer_model_dropdown": "Select the model for enhancing face details.",
    "ff3_face_enhancer_blend_slider": "Control how much the face enhancement blends with the original.",
    "ff3_frame_enhancer_model_dropdown": "Pick the model for frame enhancement.",
    "ff3_frame_enhancer_blend_slider": "Adjust the blending strength for frame enhancements.",
    "ff3_style_change_model_dropdown": "Choose a style change model for the output.",
    "ff3_style_changer_target_radio": "Specify whether to style the source or target media.",
    "ff3_temp_frame_format_dropdown": "Set the format for temporary frames during processing.",
    "ff3_temp_frame_quality_slider": "Adjust the quality of temporary frames.",
    "ff3_output_path_textbox": "Define the output file path or directory (use .mp4 for videos).",
    "ff3_output_image_quality_slider": "Set the quality level for output images.",
    "ff3_output_video_encoder_dropdown": "Choose the encoding format for the output video.",
    "ff3_output_video_quality_slider": "Adjust the quality level for the output video.",
    "ff3_source_file": "Upload audio/image files for the first source person.",
    "ff3_source_file_2": "Upload audio/image files for the second source person.",
    "ff3_target_path": "Enter a URL or remote path to the target media.",
    "ff3_target_file": "Upload a local video or image as the target for face swapping.",
    "ff3_trim_frame_start_slider": "Set the start time for processing the target video.",
    "ff3_trim_frame_end_slider": "Set the end time for processing the target video.",
    "ff3_face_recognition_dropdown": "Choose a recognition mode: single face or multiple faces.",
    "ff3_reference_face_position_gallery": "Select the reference face from the source image.",
    "ff3_reference_face_distance_slider": "Set the estimated distance to the reference face.",
    "ff3_face_analyser_direction_dropdown": "Analyze the orientation of the detected face.",
    "ff3_face_analyser_age_dropdown": "Estimate the age of detected faces.",
    "ff3_face_analyser_gender_dropdown": "Determine the gender of detected faces.",
    "ff3_webcam_start_button": "Activate the webcam for live face processing.",
    "ff3_webcam_stop_button": "Deactivate the webcam.",
    "ff3_age_modifier_model_dropdown": "Select a model for modifying age appearance.",
    "ff3_age_modifier_direction_slider": "Adjust how much the age modification is applied.",
    "ff3_expression_restorer_model_dropdown": "Choose a model to restore facial expressions.",
    "ff3_expression_restorer_factor_slider": "Set the strength of the expression restoration.",
    "ff3_face_debugger_items_checkbox_group": "Toggle display of debugging information for faces.",
    "ff3_face_editor_model_dropdown": "Pick a model for detailed face editing.",
    "ff3_face_editor_eyebrow_direction_slider": "Adjust the angle of the eyebrows.",
    "ff3_face_editor_eye_gaze_horizontal_slider": "Shift the horizontal eye gaze direction.",
    "ff3_face_editor_eye_gaze_vertical_slider": "Shift the vertical eye gaze direction.",
    "ff3_face_editor_eye_open_ratio_slider": "Modify the openness of the eyes.",
    "ff3_face_editor_lip_open_ratio_slider": "Adjust how open the lips appear.",
    "ff3_face_editor_mouth_grim_slider": "Control the grimace level of the mouth.",
    "ff3_face_editor_mouth_pout_slider": "Adjust how much the mouth is pouting.",
    "ff3_face_editor_mouth_purse_slider": "Control the purse level of the lips.",
    "ff3_face_editor_mouth_smile_slider": "Set the smile intensity.",
    "ff3_face_editor_mouth_position_horizontal_slider": "Shift the mouth horizontally.",
    "ff3_face_editor_mouth_position_vertical_slider": "Shift the mouth vertically.",
    "ff3_face_editor_head_pitch_slider": "Tilt the head forward or backward.",
    "ff3_face_editor_head_yaw_slider": "Turn the head left or right.",
    "ff3_face_editor_head_roll_slider": "Roll the head sideways.",
    "ff3_face_swapper_pixel_boost_dropdown": "Enhance pixel quality during face swapping.",
    "ff3_frame_colorizer_model_dropdown": "Choose a model to add color to frames.",
    "ff3_frame_colorizer_size_dropdown": "Select the size of colorized frames.",
    "ff3_frame_colorizer_blend_slider": "Adjust the intensity of the colorization effect.",
    "ff3_lip_syncer_model_dropdown": "Choose a model for syncing lip movements.",
    "ff3_style_changer_model_dropdown": "Pick a model to apply style changes.",
    "ff3_style_target_radio": "Specify whether the style applies to the source or target.",
    "ff3_output_video_preset_dropdown": "Select a preset for output video settings.",
    "ff3_output_video_resolution_dropdown": "Set the resolution for the output video.",
    "ff3_output_video_fps_slider": "Adjust the frame rate for the output video.",
    "ff3_output_image": "Generate and display the output image.",
    "ff3_output_video": "Generate and display the output video.",
    "ff3_source_audio": "Select an audio file for processing.",
    "ff3_source_image": "Choose an image file as the source.",
    "ff3_source_image_2": "Choose an additional image file as the source.",
    "ff3_target_image": "Upload an image to serve as the target.",
    "ff3_target_video": "Upload a video to serve as the target.",
    "ff3_sync_video_lip": "Synchronize the video with lip movements.",
    "ff3_preview_image": "Preview the processed image before finalizing.",
    "ff3_preview_frame_row": "View a row of preview frames.",
    "ff3_preview_frame_back_five_button": "Jump back five frames in the preview.",
    "ff3_preview_frame_back_button": "Move back one frame in the preview.",
    "ff3_preview_frame_slider": "Slide to adjust the current frame in the preview.",
    "ff3_preview_frame_forward_button": "Move forward one frame in the preview.",
    "ff3_preview_frame_forward_five_button": "Jump forward five frames in the preview.",
    "ff3_face_selector_mode_dropdown": "Choose the mode for selecting faces.",
    "ff3_add_reference_face_button": "Add a new reference face to the selection for person 1.",
    "ff3_add_reference_face_button_2": "Add a new reference face to the selection for person 2.",
    "ff3_remove_reference_faces_button": "Remove selected reference faces for person 1.",
    "ff3_remove_reference_faces_button_2": "Remove selected reference faces for person 2.",
    "ff3_reference_faces_selection_gallery": "Selected reference faces for the first person.",
    "ff3_reference_faces_selection_gallery_2": "Selected reference faces for the second person.",
    "ff3_face_selector_order_dropdown": "Set the order for selecting faces.",
    "ff3_face_selector_gender_dropdown": "Filter face selection by gender.",
    "ff3_face_selector_race_dropdown": "Filter face selection by race.",
    "ff3_face_selector_age_range_start_slider": "Set the starting range for age filtering.",
    "ff3_face_selector_age_range_end_slider": "Set the ending range for age filtering.",
    "ff3_face_mask_types_checkbox_group": "Select types of face masks to apply.",
    "ff3_face_mask_regions_checkbox_group": "Choose specific regions for face masking.",
    "ff3_face_mask_blur_slider": "Control the blur intensity of the face mask.",
    "ff3_mask_disable_button": "Disable the current face mask.",
    "ff3_mask_enable_button": "Enable the selected face mask.",
    "ff3_mask_clear_button": "Clear all face masks.",
    "ff3_face_mask_padding_top_slider": "Adjust padding at the top of the face mask.",
    "ff3_face_mask_padding_right_slider": "Adjust padding at the right of the face mask.",
    "ff3_face_mask_padding_bottom_slider": "Adjust padding at the bottom of the face mask.",
    "ff3_face_mask_padding_left_slider": "Adjust padding at the left of the face mask.",
    "ff3_face_detector_model_dropdown": "Choose the model for detecting faces.",
    "ff3_face_detector_size_dropdown": "Set the size for face detection.",
    "ff3_face_detector_angles_checkbox_group": "Select the angles for face detection.",
    "ff3_face_detector_score_slider": "Adjust the threshold score for face detection.",
    "ff3_face_landmarker_model_dropdown": "Pick a model for face landmarking.",
    "ff3_face_landmarker_score_slider": "Set the confidence threshold for face landmarks.",
    "ff3_benchmark_runs_checkbox_group": "Select benchmarking options.",
    "ff3_benchmark_cycles_slider": "Set the number of benchmark cycles to run."
};

onUiUpdate(function () {
    if (!hints_set) {
        for (const [id, hint] of Object.entries(hint_dict)) {
            // Get the element
            const element = gradioApp().getElementById(id);
            // If the element exists
            if (element) {
                // Set the title attribute
                element.setAttribute("title", hint);
                // Set the title of all the element's children as well
                for (const child of element.children) {
                    child.setAttribute("title", hint);
                }
                hints_set = true;
            }
        }
        // // Get all elements in gradioApp that start with ff_
        // const elements = gradioApp().querySelectorAll("[id^='ff_']");
        // // For each element
        // for (const element of elements) {
        //     // If the element has no title attribute
        //     if (!element.hasAttribute("title")) {
        //         // Set the title attribute
        //         element.setAttribute("title", "No hint available");
        //     }
        //     missingHints.push(element.id);
        // }

    }

});