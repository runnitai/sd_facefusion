let hints_set = false;

const hint_dict = {
    "ff_start": "Start Processing all jobs in queue",
    "ff_clear": "Cancel Processing of all jobs in queue",
    "frame_processors_checkbox_group": "Select frame processors",
    "face_swapper_model_dropdown": "Select face swapper model",
    "face_enhancer_model_dropdown": "Select face enhancer model",
    "face_enhancer_blend_slider": "Adjust face enhancer blend amount",
    "frame_enhancer_model_dropdown": "Select frame enhancer model",
    "frame_enhancer_blend_slider": "Adjust frame enhancer blend amount",
    "execution_thread_count_slider": "Adjust execution thread count (Appx total amount of VRAM * 2)",
    "temp_frame_format_dropdown": "Select temporary frame format",
    "temp_frame_quality_slider": "Adjust temporary frame quality",
    "output_path_textbox": "Set output path. Can be a custom filename or a directory. Be sure to use .mp4 extension for video files.",
    "output_image_quality_slider": "Adjust output image quality",
    "output_video_encoder_dropdown": "Select output video encoder",
    "output_video_quality_slider": "Adjust output video quality",
    "queueTable": "Job Queue. Select a job to remove it.",
    "ff_enqueue": "Add job to queue",
    "ff_remove_last": "Remove selected element from job queue",
    "ff_clear_queue": "Clear job queue",
    "ff_job_queue_options_checkbox_group": "Select job queue options",
    "ff_source_file": "Select an image to use as the source face",
    "ff_target_path": "A URL which points to a youtube video, online video, or online image; or a remote path (/mnt/private/images...)",
    "ff_target_file": "Drop or select a video or image to apply the face to from your local computer",
    "ff_trim_frame_start_slider": "Adjust the start time of the target video",
    "ff_trim_frame_end_slider": "Adjust the end time of the target video",
    "ff_face_recognition_dropdown": "Select face recognition mode. Use 'reference' for a single face, 'many' to replace all faces.",
    "ff_reference_face_position_gallery": "Select a face from the source image to use as the reference face",
    "ff_reference_face_distance_slider": "Approximate distance to the reference face",
    "ff_face_analyser_direction_dropdown": "Select face analyser direction",
    "ff_face_analyser_age_dropdown": "Select face analyser age",
    "ff_face_analyser_gender_dropdown": "Select face analyzer gender",
    "ff_webcam_start_button": "Start webcam",
    "ff_webcam_stop_button": "Stop webcam",
}

// Add DOMContentLoaded listener to the document
document.addEventListener('DOMContentLoaded', function() {
    // Loop through hint_dict
    for (const [id, hint] of Object.entries(hint_dict)) {
        // Get the element
        const element = gradioApp().getElementById(id);
        // If the element exists
        if (element) {
            // Set the title attribute
            element.setAttribute("title", hint);
            // Set the title of all of the element's children as well
            for (const child of element.children) {
                child.setAttribute("title", hint);
            }
        }
    }
});

onUiUpdate(function () {
		if (!hints_set) {
			for (const [id, hint] of Object.entries(hint_dict)) {
				// Get the element
				const element = gradioApp().getElementById(id);
				// If the element exists
				if (element) {
					// Set the title attribute
					element.setAttribute("title", hint);
					// Set the title of all of the element's children as well
					for (const child of element.children) {
						child.setAttribute("title", hint);
					}
				}
			}
			hints_set = true;
		}
});