from typing import List, Optional, Tuple, Dict
import threading

import gradio
from gradio import SelectData

import facefusion.choices
from facefusion import wording, state_manager, logger
from facefusion.common_helper import calc_float_step, calc_int_step
from facefusion.face_analyser import get_many_faces
from facefusion.face_selector import sort_and_filter_faces, current_sort_values, calc_face_distance
from facefusion.face_store import clear_reference_faces, clear_static_faces, get_reference_faces
from facefusion.filesystem import is_image, is_video
from facefusion.processors.core import get_processors_modules
from facefusion.typing import FaceSelectorMode, VisionFrame, Race, Gender, FaceSelectorOrder, FaceReference, Face
from facefusion.uis.components.face_masker import clear_mask_times
from facefusion.uis.core import get_ui_component, register_ui_component, get_ui_components
# from gradio_rangeslider import RangeSlider
from facefusion.uis.typing import ComponentOptions
from facefusion.uis.ui_helper import convert_str_none
from facefusion.vision import get_video_frame, normalize_frame_color, read_static_image, detect_video_fps, \
    count_video_frame_total
from facefusion.video_face_index import VIDEO_FACE_INDEX

FACE_SELECTOR_MODE_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SELECTOR_ORDER_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SELECTOR_GENDER_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SELECTOR_RACE_DROPDOWN: Optional[gradio.Dropdown] = None
# FACE_SELECTOR_AGE_RANGE_SLIDER: Optional[RangeSlider] = None
FACE_SELECTOR_AGE_RANGE_START_SLIDER: Optional[gradio.Slider] = None
FACE_SELECTOR_AGE_RANGE_END_SLIDER: Optional[gradio.Slider] = None
REFERENCE_FACE_POSITION_GALLERY: Optional[gradio.Gallery] = None
REFERENCE_FACE_POSITION_GALLERY_2: Optional[gradio.Gallery] = None
REFERENCE_FACE_DISTANCE_SLIDER: Optional[gradio.Slider] = None
ADD_REFERENCE_FACE_BUTTON: Optional[gradio.Button] = None
REMOVE_REFERENCE_FACE_BUTTON: Optional[gradio.Button] = None
REFERENCE_FACES_SELECTION_GALLERY: Optional[gradio.Gallery] = None
ADD_REFERENCE_FACE_BUTTON_2: Optional[gradio.Button] = None
REMOVE_REFERENCE_FACE_BUTTON_2: Optional[gradio.Button] = None
REFERENCE_FACES_SELECTION_GALLERY_2: Optional[gradio.Gallery] = None
FACE_SELECTOR_GROUP: Optional[gradio.Group] = None

# Face Cache Components
INDEX_VIDEO_BUTTON: Optional[gradio.Button] = None
INDEX_STATUS_TEXT: Optional[gradio.Textbox] = None
FIND_UNMATCHED_BUTTON: Optional[gradio.Button] = None
ALL_UNMATCHED_FACES_GALLERY: Optional[gradio.Gallery] = None
FACE_NUMBER_INPUT: Optional[gradio.Number] = None
IGNORE_FACE_BUTTON: Optional[gradio.Button] = None
CLEAR_CACHE_BUTTON: Optional[gradio.Button] = None
CACHE_INFO_TEXT: Optional[gradio.Textbox] = None

# Reference frame and faces
current_reference_faces = []
current_reference_frames = []

# Stores the selected reference face data
current_selected_faces = []
current_selected_faces_2 = []

# Indices for the face selector and selections
selector_face_index = -1
selected_face_index = -1
selected_face_index_2 = -1

# Face cache state
all_unmatched_faces: List[Tuple[int, int, Face]] = []  # [(frame_number, face_index, face)]
gallery_to_face_mapping: Dict[int, Tuple[int, int]] = {}  # {gallery_index: (frame_number, face_index)}
current_selected_gallery_index: Optional[int] = None
indexing_in_progress = False
_indexing_lock = threading.Lock()


def render() -> None:
    global FACE_SELECTOR_MODE_DROPDOWN
    global FACE_SELECTOR_ORDER_DROPDOWN
    global FACE_SELECTOR_GENDER_DROPDOWN
    global FACE_SELECTOR_RACE_DROPDOWN
    # global FACE_SELECTOR_AGE_RANGE_SLIDER
    global FACE_SELECTOR_AGE_RANGE_START_SLIDER
    global FACE_SELECTOR_AGE_RANGE_END_SLIDER
    global REFERENCE_FACE_POSITION_GALLERY
    global REFERENCE_FACE_DISTANCE_SLIDER
    global ADD_REFERENCE_FACE_BUTTON
    global REMOVE_REFERENCE_FACE_BUTTON
    global REFERENCE_FACES_SELECTION_GALLERY
    global REFERENCE_FACES_SELECTION_GALLERY_2
    global ADD_REFERENCE_FACE_BUTTON_2
    global REMOVE_REFERENCE_FACE_BUTTON_2
    global FACE_SELECTOR_GROUP
    
    # Face Cache Components
    global INDEX_VIDEO_BUTTON, INDEX_STATUS_TEXT
    global FIND_UNMATCHED_BUTTON, ALL_UNMATCHED_FACES_GALLERY, FACE_NUMBER_INPUT
    global IGNORE_FACE_BUTTON, CLEAR_CACHE_BUTTON, CACHE_INFO_TEXT

    reference_face_gallery_options: ComponentOptions = \
        {
            'label': wording.get('uis.reference_face_gallery'),
            'object_fit': 'cover',
            'columns': 8,
            'allow_preview': False,
            'visible': 'reference' in state_manager.get_item('face_selector_mode')
        }
    if is_image(state_manager.get_item('target_path')):
        reference_frame = read_static_image(state_manager.get_item('target_path'))
        reference_face_gallery_options['value'] = extract_gallery_frames(reference_frame)
    if is_video(state_manager.get_item('target_path')):
        reference_frame = get_video_frame(state_manager.get_item('target_path'),
                                          state_manager.get_item('reference_frame_number'))
        reference_face_gallery_options['value'] = extract_gallery_frames(reference_frame)
    non_face_processors = ['frame_colorizer', 'frame_enhancer', 'style_transfer']
    # Make the group visible if any face processor is selected
    show_group = False
    for processor in state_manager.get_item('processors'):
        if processor not in non_face_processors:
            show_group = True
            break
    with gradio.Group(visible=show_group) as FACE_SELECTOR_GROUP:
        FACE_SELECTOR_MODE_DROPDOWN = gradio.Dropdown(
            label=wording.get('uis.face_selector_mode_dropdown'),
            choices=facefusion.choices.face_selector_modes,
            value=state_manager.get_item('face_selector_mode')
        )

        # Face Cache Management Section
        with gradio.Group() as face_cache_group:
            gradio.Markdown("### Face Cache Management")
            
            # First row: Cache info and status
            with gradio.Row():
                INDEX_STATUS_TEXT = gradio.Textbox(
                    label="Cache Status",
                    value="Ready to index video",
                    interactive=False,
                    elem_id="index_status_text"
                )
                
                CACHE_INFO_TEXT = gradio.Textbox(
                    label="Cache Info",
                    value="No cache information available",
                    interactive=False,
                    elem_id="cache_info_text"
                )
            
            # Second row: Action buttons
            with gradio.Row():
                INDEX_VIDEO_BUTTON = gradio.Button(
                    value="🔍 Index Video Faces",
                    variant="primary",
                    elem_id="index_video_button"
                )
                
                CLEAR_CACHE_BUTTON = gradio.Button(
                    value="🗑️ Clear Cache",
                    variant="secondary",
                    elem_id="clear_cache_button"
                )
            
            with gradio.Row():
                FIND_UNMATCHED_BUTTON = gradio.Button(
                    value="🔍 Find All Unmatched Faces",
                    variant="secondary",
                    elem_id="find_unmatched_button"
                )
            
            with gradio.Row():
                ALL_UNMATCHED_FACES_GALLERY = gradio.Gallery(
                    label="Unmatched Faces (Click to select and jump to frame)",
                    object_fit="cover",
                    columns=8,
                    allow_preview=False,
                    visible=False,
                    elem_id="all_unmatched_faces_gallery"
                )
            
            with gradio.Row():
                with gradio.Column():
                    gradio.Markdown("**Instructions:** 1) Index video, 2) Find unmatched faces, 3) Click face to jump to frame, 4) Choose action")
                    
                    # Fallback face selection by number
                    FACE_NUMBER_INPUT = gradio.Number(
                        label="Or select face by number (1, 2, 3...)",
                        value=None,
                        minimum=1,
                        step=1,
                        visible=False,
                        elem_id="face_number_input"
                    )
                    
                    with gradio.Row():
                        IGNORE_FACE_BUTTON = gradio.Button(
                            value="🚫 Ignore Face Forever",
                            variant="secondary",
                            visible=False,
                            elem_id="ignore_face_button"
                        )

        with gradio.Row():
            with gradio.Column(scale=0, min_width="33"):
                ADD_REFERENCE_FACE_BUTTON = gradio.Button(
                    value="+1",
                    elem_id='ff_add_reference_face_button'
                )
                ADD_REFERENCE_FACE_BUTTON_2 = gradio.Button(
                    value="+2",
                    elem_id='ff_add_reference_face_button'
                )
            with gradio.Column():
                REFERENCE_FACE_POSITION_GALLERY = gradio.Gallery(**reference_face_gallery_options,
                                                                 elem_id='ff_reference_face_position_gallery')
        with gradio.Row():
            REMOVE_REFERENCE_FACE_BUTTON = gradio.Button(
                value="-",
                variant='secondary',
                elem_id='ff_remove_reference_faces_button',
                elem_classes=['remove_reference_faces_button']
            )
            REFERENCE_FACES_SELECTION_GALLERY = gradio.Gallery(
                label="Selected Faces (Source 1)",
                object_fit='cover',
                columns=8,
                allow_preview=False,
                visible='reference' in state_manager.get_item('face_selector_mode'),
                elem_id='ff_reference_faces_selection_gallery'
            )
        with gradio.Row():
            REMOVE_REFERENCE_FACE_BUTTON_2 = gradio.Button(
                value="-",
                variant='secondary',
                elem_id='ff_remove_reference_faces_button_2',
                elem_classes=['remove_reference_faces_button']
            )
            REFERENCE_FACES_SELECTION_GALLERY_2 = gradio.Gallery(
                label="Selected Faces (Source 2)",
                object_fit='cover',
                columns=8,
                allow_preview=False,
                visible='reference' in state_manager.get_item('face_selector_mode'),
                elem_id='ff_reference_faces_selection_gallery'
            )
        with gradio.Row():
            FACE_SELECTOR_ORDER_DROPDOWN = gradio.Dropdown(
                label=wording.get('uis.face_selector_order_dropdown'),
                choices=facefusion.choices.face_selector_orders,
                value=state_manager.get_item('face_selector_order')
            )
            FACE_SELECTOR_GENDER_DROPDOWN = gradio.Dropdown(
                label=wording.get('uis.face_selector_gender_dropdown'),
                choices=['none'] + facefusion.choices.face_selector_genders,
                value=state_manager.get_item('face_selector_gender') or 'none'
            )
            FACE_SELECTOR_RACE_DROPDOWN = gradio.Dropdown(
                label=wording.get('uis.face_selector_race_dropdown'),
                choices=['none'] + facefusion.choices.face_selector_races,
                value=state_manager.get_item('face_selector_race') or 'none'
            )
        with gradio.Row():
            face_selector_age_start = state_manager.get_item('face_selector_age_start') or \
                                      facefusion.choices.face_selector_age_range[0]
            face_selector_age_end = state_manager.get_item('face_selector_age_end') or \
                                    facefusion.choices.face_selector_age_range[-1]
            with gradio.Row():
                FACE_SELECTOR_AGE_RANGE_START_SLIDER = gradio.Slider(
                    label=wording.get('uis.face_selector_age_start_slider'),
                    value=face_selector_age_start,
                    step=calc_int_step(facefusion.choices.face_selector_age_range),
                    minimum=facefusion.choices.face_selector_age_range[0],
                    maximum=facefusion.choices.face_selector_age_range[-1]
                )
                FACE_SELECTOR_AGE_RANGE_END_SLIDER = gradio.Slider(
                    label=wording.get('uis.face_selector_age_end_slider'),
                    value=face_selector_age_end,
                    step=calc_int_step(facefusion.choices.face_selector_age_range),
                    minimum=facefusion.choices.face_selector_age_range[0],
                    maximum=facefusion.choices.face_selector_age_range[-1]
                )
            REFERENCE_FACE_DISTANCE_SLIDER = gradio.Slider(
                label=wording.get('uis.reference_face_distance_slider'),
                value=state_manager.get_item('reference_face_distance'),
                step=calc_float_step(facefusion.choices.reference_face_distance_range),
                minimum=facefusion.choices.reference_face_distance_range[0],
                maximum=facefusion.choices.reference_face_distance_range[-1],
                visible='reference' in state_manager.get_item('face_selector_mode')
            )
    register_ui_component('face_selector_mode_dropdown', FACE_SELECTOR_MODE_DROPDOWN)
    register_ui_component('face_selector_order_dropdown', FACE_SELECTOR_ORDER_DROPDOWN)
    register_ui_component('face_selector_gender_dropdown', FACE_SELECTOR_GENDER_DROPDOWN)
    register_ui_component('face_selector_race_dropdown', FACE_SELECTOR_RACE_DROPDOWN)
    # register_ui_component('face_selector_age_range_slider', FACE_SELECTOR_AGE_RANGE_SLIDER)
    register_ui_component('face_selector_age_range_start_slider', FACE_SELECTOR_AGE_RANGE_START_SLIDER)
    register_ui_component('face_selector_age_range_end_slider', FACE_SELECTOR_AGE_RANGE_END_SLIDER)
    register_ui_component('reference_face_position_gallery', REFERENCE_FACE_POSITION_GALLERY)
    register_ui_component('reference_faces_selection_gallery', REFERENCE_FACES_SELECTION_GALLERY)
    register_ui_component('reference_faces_selection_gallery_2', REFERENCE_FACES_SELECTION_GALLERY_2)
    register_ui_component('reference_face_distance_slider', REFERENCE_FACE_DISTANCE_SLIDER)
    register_ui_component('add_reference_face_button', ADD_REFERENCE_FACE_BUTTON)
    register_ui_component('remove_reference_faces_button', REMOVE_REFERENCE_FACE_BUTTON)
    register_ui_component('remove_reference_faces_button_2', REMOVE_REFERENCE_FACE_BUTTON_2)
    register_ui_component('reference_faces_selection_gallery', REFERENCE_FACES_SELECTION_GALLERY)
    register_ui_component('add_reference_face_button_2', ADD_REFERENCE_FACE_BUTTON_2)
    register_ui_component('face_selector_group', FACE_SELECTOR_GROUP)
    
    # Register face cache components
    register_ui_component('index_video_button', INDEX_VIDEO_BUTTON)
    register_ui_component('index_status_text', INDEX_STATUS_TEXT)
    register_ui_component('find_unmatched_button', FIND_UNMATCHED_BUTTON)
    register_ui_component('all_unmatched_faces_gallery', ALL_UNMATCHED_FACES_GALLERY)
    register_ui_component('face_number_input', FACE_NUMBER_INPUT)
    register_ui_component('ignore_face_button', IGNORE_FACE_BUTTON)
    register_ui_component('clear_cache_button', CLEAR_CACHE_BUTTON)
    register_ui_component('cache_info_text', CACHE_INFO_TEXT)


def listen() -> None:
    FACE_SELECTOR_MODE_DROPDOWN.select(
        update_face_selector_mode,
        inputs=FACE_SELECTOR_MODE_DROPDOWN,
        outputs=[
            REFERENCE_FACE_POSITION_GALLERY,
            REFERENCE_FACES_SELECTION_GALLERY,
            REFERENCE_FACES_SELECTION_GALLERY_2,
            REFERENCE_FACE_DISTANCE_SLIDER
        ]
    )

    REFERENCE_FACE_POSITION_GALLERY.select(update_selector_face_index)
    REFERENCE_FACES_SELECTION_GALLERY.select(update_selected_face_index)
    REFERENCE_FACES_SELECTION_GALLERY_2.select(update_selected_face_index_2)

    REFERENCE_FACE_DISTANCE_SLIDER.change(
        update_reference_face_distance,
        inputs=REFERENCE_FACE_DISTANCE_SLIDER,
        outputs=[]
    )

    ADD_REFERENCE_FACE_BUTTON.click(
        add_reference_face,
        inputs=[
            REFERENCE_FACE_POSITION_GALLERY,
            REFERENCE_FACES_SELECTION_GALLERY,
            get_ui_component('preview_frame_slider')
        ],
        outputs=[
            REFERENCE_FACES_SELECTION_GALLERY,
            get_ui_component('preview_image'),
            get_ui_component('mask_enable_button'),
            get_ui_component('mask_disable_button')
        ]
    )

    ADD_REFERENCE_FACE_BUTTON_2.click(
        add_reference_face_2,
        inputs=[
            REFERENCE_FACE_POSITION_GALLERY,
            REFERENCE_FACES_SELECTION_GALLERY_2,
            get_ui_component('preview_frame_slider')
        ],
        outputs=[
            REFERENCE_FACES_SELECTION_GALLERY_2,
            get_ui_component('preview_image'),
            get_ui_component('mask_enable_button'),
            get_ui_component('mask_disable_button')
        ]
    )

    REMOVE_REFERENCE_FACE_BUTTON.click(
        remove_reference_face,
        inputs=[
            REFERENCE_FACES_SELECTION_GALLERY,
            get_ui_component('preview_frame_slider')
        ],
        outputs=[
            REFERENCE_FACES_SELECTION_GALLERY,
            get_ui_component('preview_image'),
            get_ui_component('mask_enable_button'),
            get_ui_component('mask_disable_button')
        ]
    )

    REMOVE_REFERENCE_FACE_BUTTON_2.click(
        remove_reference_face_2,
        inputs=[
            REFERENCE_FACES_SELECTION_GALLERY_2,
            get_ui_component('preview_frame_slider')
        ],
        outputs=[
            REFERENCE_FACES_SELECTION_GALLERY_2,
            get_ui_component('preview_image'),
            get_ui_component('mask_enable_button'),
            get_ui_component('mask_disable_button')
        ]
    )

    # Face Cache Event Listeners
    if INDEX_VIDEO_BUTTON and INDEX_STATUS_TEXT:
        INDEX_VIDEO_BUTTON.click(
            index_video_faces,
            outputs=[INDEX_STATUS_TEXT, CACHE_INFO_TEXT]
        )
    
    if FIND_UNMATCHED_BUTTON:
        FIND_UNMATCHED_BUTTON.click(
            find_and_display_all_unmatched_faces,
            outputs=[
                ALL_UNMATCHED_FACES_GALLERY, 
                INDEX_STATUS_TEXT,
                REFERENCE_FACE_POSITION_GALLERY,
                FACE_NUMBER_INPUT,
                IGNORE_FACE_BUTTON
            ]
        )
    
    if ALL_UNMATCHED_FACES_GALLERY:
        ALL_UNMATCHED_FACES_GALLERY.select(
            select_unmatched_face_and_jump,
            outputs=[
                INDEX_STATUS_TEXT,
                get_ui_component('preview_frame_slider'),
                get_ui_component('preview_image'),
                REFERENCE_FACE_POSITION_GALLERY
            ]
        )
    
    if FACE_NUMBER_INPUT:
        FACE_NUMBER_INPUT.change(
            select_face_by_number,
            inputs=[FACE_NUMBER_INPUT],
            outputs=[INDEX_STATUS_TEXT]
        )
    
    if IGNORE_FACE_BUTTON:
        IGNORE_FACE_BUTTON.click(
            ignore_selected_face,
            outputs=[INDEX_STATUS_TEXT, ALL_UNMATCHED_FACES_GALLERY]
        )
    
    if CLEAR_CACHE_BUTTON:
        CLEAR_CACHE_BUTTON.click(
            clear_video_cache,
            outputs=[INDEX_STATUS_TEXT, CACHE_INFO_TEXT, ALL_UNMATCHED_FACES_GALLERY]
        )

    get_ui_component('preview_frame_slider').release(
        update_reference_frame_number,
        inputs=get_ui_component('preview_frame_slider'),
        outputs=[
            REFERENCE_FACE_POSITION_GALLERY,
            REFERENCE_FACES_SELECTION_GALLERY,
            REFERENCE_FACES_SELECTION_GALLERY_2
        ]
    )

    get_ui_component('preview_frame_slider').release(
        update_reference_position_gallery,
        outputs=[
            REFERENCE_FACE_POSITION_GALLERY,
            REFERENCE_FACES_SELECTION_GALLERY,
            REFERENCE_FACES_SELECTION_GALLERY_2
        ]
    )

    get_ui_component('preview_frame_back_button').click(
        reference_frame_back,
        inputs=get_ui_component('preview_frame_slider'),
        outputs=[
            get_ui_component('preview_frame_slider'),
            REFERENCE_FACE_POSITION_GALLERY,
            REFERENCE_FACES_SELECTION_GALLERY
        ]
    )

    get_ui_component('preview_frame_forward_button').click(
        reference_frame_forward,
        inputs=get_ui_component('preview_frame_slider'),
        outputs=[
            get_ui_component('preview_frame_slider'),
            REFERENCE_FACE_POSITION_GALLERY,
            REFERENCE_FACES_SELECTION_GALLERY
        ]
    )

    get_ui_component('preview_frame_back_five_button').click(
        reference_frame_back_five,
        inputs=get_ui_component('preview_frame_slider'),
        outputs=[
            get_ui_component('preview_frame_slider'),
            REFERENCE_FACE_POSITION_GALLERY,
            REFERENCE_FACES_SELECTION_GALLERY
        ]
    )

    get_ui_component('preview_frame_forward_five_button').click(
        reference_frame_forward_five,
        inputs=get_ui_component('preview_frame_slider'),
        outputs=[
            get_ui_component('preview_frame_slider'),
            REFERENCE_FACE_POSITION_GALLERY,
            REFERENCE_FACES_SELECTION_GALLERY
        ]
    )

    for ui_component in get_ui_components(['target_image', 'target_video']):
        for method in ['upload', 'change', 'clear']:
            getattr(ui_component, method)(
                update_reference_face_position,
                outputs=[]
            )
            getattr(ui_component, method)(
                update_reference_position_gallery,
                outputs=[
                    REFERENCE_FACE_POSITION_GALLERY,
                    REFERENCE_FACES_SELECTION_GALLERY,
                    REFERENCE_FACES_SELECTION_GALLERY_2
                ]
            )
            # Auto-check cache status when target changes
            getattr(ui_component, method)(
                check_and_update_cache_status,
                outputs=[INDEX_STATUS_TEXT, CACHE_INFO_TEXT]
            )

    for ui_component in get_ui_components([
        'face_detector_model_dropdown',
        'face_detector_size_dropdown',
        'face_detector_angles_checkbox_group'
    ]):
        ui_component.change(
            clear_and_update_reference_position_gallery,
            outputs=[
                REFERENCE_FACE_POSITION_GALLERY,
                REFERENCE_FACES_SELECTION_GALLERY,
                REFERENCE_FACES_SELECTION_GALLERY_2
            ]
        )

    face_detector_score_slider = get_ui_component('face_detector_score_slider')
    if face_detector_score_slider:
        face_detector_score_slider.release(
            clear_and_update_reference_position_gallery,
            outputs=[
                REFERENCE_FACE_POSITION_GALLERY,
                REFERENCE_FACES_SELECTION_GALLERY,
                REFERENCE_FACES_SELECTION_GALLERY_2
            ]
        )
    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(
            toggle_group,
            inputs=processors_checkbox_group,
            outputs=[FACE_SELECTOR_GROUP]
        )


def toggle_group(processors: List[str]) -> gradio.update:
    all_processors = get_processors_modules()
    all_face_processor_names = [processor.display_name for processor in all_processors if processor.is_face_processor]
    # Make the group visible if any face processor is selected
    for processor in processors:
        if processor in all_face_processor_names:
            return gradio.update(visible=True)
    return gradio.update(visible=False)


def update_face_selector_mode(face_selector_mode: FaceSelectorMode) -> Tuple[gradio.Gallery, gradio.update]:
    state_manager.set_item('face_selector_mode', face_selector_mode)
    if face_selector_mode == 'many':
        return gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=False)
    if face_selector_mode == 'one':
        return gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=False)
    if face_selector_mode == 'reference':
        return gradio.update(visible=True), gradio.update(visible=True), gradio.update(visible=True), gradio.update(visible=True)
    return None


def update_face_selector_order(face_analyser_order: FaceSelectorOrder) -> gradio.Gallery:
    state_manager.set_item('face_selector_order', convert_str_none(face_analyser_order))
    return update_reference_position_gallery()


def update_face_selector_gender(face_selector_gender: Gender) -> gradio.Gallery:
    state_manager.set_item('face_selector_gender', convert_str_none(face_selector_gender))
    return update_reference_position_gallery()


def update_face_selector_race(face_selector_race: Race) -> gradio.Gallery:
    state_manager.set_item('face_selector_race', convert_str_none(face_selector_race))
    return update_reference_position_gallery()


def update_face_selector_age_range(face_selector_age_range: Tuple[float, float]) -> gradio.Gallery:
    face_selector_age_start, face_selector_age_end = face_selector_age_range
    state_manager.set_item('face_selector_age_start', int(face_selector_age_start))
    state_manager.set_item('face_selector_age_end', int(face_selector_age_end))
    return update_reference_position_gallery()


def clear_selected_faces() -> None:
    global current_selected_faces, current_selected_faces_2, selected_face_index, selected_face_index_2
    current_selected_faces = []
    current_selected_faces_2 = []
    selected_face_index = -1
    selected_face_index_2 = -1
    state_manager.set_item('reference_face_dict', {})
    state_manager.set_item('reference_face_dict_2', {})


def clear_and_update_reference_face_position(event: gradio.SelectData) -> gradio.Gallery:
    clear_reference_faces()
    clear_static_faces()
    clear_selected_faces()
    update_reference_face_position(event.index)
    return update_reference_position_gallery(), clear_mask_times()


def update_reference_face_position(reference_face_position: int = 0) -> None:
    state_manager.set_item('reference_face_position', reference_face_position)


def update_reference_face_distance(reference_face_distance: float) -> None:
    state_manager.set_item('reference_face_distance', reference_face_distance)


def update_reference_frame_number(reference_frame_number: int) -> None:
    state_manager.set_item('reference_frame_number', reference_frame_number)
    return update_reference_position_gallery()


def clear_and_update_reference_position_gallery() -> Tuple[gradio.update, gradio.update]:
    clear_reference_faces()
    clear_static_faces()
    clear_selected_faces()
    return update_reference_position_gallery()


def update_reference_position_gallery() -> Tuple[gradio.update, gradio.update]:
    gallery_frames = []
    selection_gallery = gradio.update()
    selection_gallery_2 = gradio.update()
    target_path = state_manager.get_item('target_path')
    reference_frame_number = state_manager.get_item('reference_frame_number')
    if is_image(target_path):
        reference_frame = read_static_image(target_path)
        gallery_frames = extract_gallery_frames(reference_frame)
    elif is_video(target_path):
        reference_frame = get_video_frame(target_path, reference_frame_number)
        gallery_frames = extract_gallery_frames(reference_frame)
    else:
        selection_gallery = gradio.update(value=None)
        selection_gallery_2 = gradio.update(value=None)
        state_manager.set_item('reference_face_dict', {})
        global current_selected_faces
        current_selected_faces = []
    if gallery_frames:
        return gradio.update(value=gallery_frames), selection_gallery, selection_gallery_2
    return gradio.update(value=None), selection_gallery, selection_gallery_2


def update_reference_frame_number_and_gallery(reference_frame_number) -> Tuple[gradio.update, gradio.update]:
    gallery_frames = []
    state_manager.set_item('reference_frame_number', reference_frame_number)
    selection_gallery = gradio.update()
    selection_gallery_2 = gradio.update()
    target_path = state_manager.get_item('target_path')
    if is_image(target_path):
        reference_frame = read_static_image(target_path)
        gallery_frames = extract_gallery_frames(reference_frame)
    elif is_video(target_path):
        reference_frame = get_video_frame(target_path, reference_frame_number)
        gallery_frames = extract_gallery_frames(reference_frame)
    else:
        selection_gallery = gradio.update(value=None)
        selection_gallery_2 = gradio.update(value=None)
        state_manager.set_item('reference_face_dict', {})
        global current_selected_faces
        current_selected_faces = []
    if gallery_frames:
        return gradio.update(value=reference_frame_number), gradio.update(
            value=gallery_frames), selection_gallery, selection_gallery_2
    return gradio.update(value=reference_frame_number), gradio.update(
        value=None), selection_gallery, selection_gallery_2


def extract_gallery_frames(temp_vision_frame: VisionFrame) -> List[VisionFrame]:
    global current_reference_faces
    global current_reference_frames
    gallery_vision_frames = []
    faces = sort_and_filter_faces(get_many_faces([temp_vision_frame]))
    current_reference_faces = faces
    for face in faces:
        start_x, start_y, end_x, end_y = map(int, face.bounding_box)
        padding_x = int((end_x - start_x) * 0.25)
        padding_y = int((end_y - start_y) * 0.25)
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = max(0, end_x + padding_x)
        end_y = max(0, end_y + padding_y)
        crop_vision_frame = temp_vision_frame[start_y:end_y, start_x:end_x]
        crop_vision_frame = normalize_frame_color(crop_vision_frame)
        gallery_vision_frames.append(crop_vision_frame)
    current_reference_frames = gallery_vision_frames
    return gallery_vision_frames


def append_reference_face(src_gallery, dest_gallery, reference_frame_number, src_face_idx, selector_face_index, current_selected_faces):
    dest_items = [item["name"] for item in dest_gallery]
    if src_gallery is not None and any(src_gallery):
        if len(src_gallery) <= selector_face_index:
            print("Invalid index")
            return gradio.update()

        selected_item = src_gallery[selector_face_index]
        face_data = current_reference_faces[selector_face_index]
        reference_face_dict = state_manager.get_item("reference_face_dict")

        if not reference_face_dict:
            reference_face_dict = {}

        ref_face_list = reference_face_dict.get(src_face_idx, [])

        found = any(
            entry["frame_number"] == reference_frame_number and
            entry["face_index"] == selector_face_index
            for entry in ref_face_list
        )

        if not found:
            sorts = current_sort_values()
            ref_face_list.append(FaceReference(
                frame_number=reference_frame_number,
                face_index=selector_face_index,
                sorts=sorts
            ))
            reference_face_dict[src_face_idx] = ref_face_list
            state_manager.set_item("reference_face_dict", reference_face_dict)
            current_selected_faces.append(face_data)
            dest_items.append(selected_item["name"])

        from facefusion.uis.components.preview import update_preview_image

        preview, enable_button, disable_button = update_preview_image(reference_frame_number)
        return gradio.update(value=dest_items), preview, enable_button, disable_button

    return gradio.update(), gradio.update(), gradio.update(), gradio.update()


def delete_reference_face(gallery, preview_frame_number, src_face_idx):
    global current_selected_faces, selected_face_index, current_selected_faces_2, selected_face_index_2
    selected_faces = current_selected_faces if src_face_idx == 0 else current_selected_faces_2
    selected_index = selected_face_index if src_face_idx == 0 else selected_face_index_2

    if len(gallery) <= selected_index or len(selected_faces) <= selected_index:
        print("Invalid index")
        return gradio.update()

    new_items = []
    for idx, item in enumerate(gallery):
        if idx != selected_index:
            new_items.append(item["name"])

    reference_face_dict = state_manager.get_item("reference_face_dict")
    if not reference_face_dict:
        reference_face_dict = {}

    ref_face_list = reference_face_dict.get(src_face_idx, [])

    ref_face_list = [
        entry for entry in ref_face_list
        if not (
            entry["frame_number"] == preview_frame_number and
            entry["face_index"] == selected_index
        )
    ]

    reference_face_dict[src_face_idx] = ref_face_list
    state_manager.set_item("reference_face_dict", reference_face_dict)
    selected_faces.pop(selected_index)

    from facefusion.uis.components.preview import update_preview_image
    preview_image, enable_button, disable_button = update_preview_image(preview_frame_number)
    return gradio.update(value=new_items), preview_image, enable_button, disable_button


def add_reference_face(src_gallery, dest_gallery, reference_frame_number):
    """Add reference face - handles both regular selection and unmatched face assignment"""
    global current_selected_gallery_index, gallery_to_face_mapping
    
    # Check if we have an unmatched face selected
    if current_selected_gallery_index is not None and current_selected_gallery_index in gallery_to_face_mapping:
        return assign_unmatched_face_to_source(0)
    
    # Regular reference face selection
    return append_reference_face(
        src_gallery, dest_gallery, reference_frame_number,
        src_face_idx=0,
        selector_face_index=selector_face_index,
        current_selected_faces=current_selected_faces
    )


def remove_reference_face(gallery, preview_frame_number):
    return delete_reference_face(
        gallery, preview_frame_number, 0
    )


def add_reference_face_2(src_gallery, dest_gallery, reference_frame_number):
    """Add reference face 2 - handles both regular selection and unmatched face assignment"""
    global current_selected_gallery_index, gallery_to_face_mapping
    
    # Check if we have an unmatched face selected
    if current_selected_gallery_index is not None and current_selected_gallery_index in gallery_to_face_mapping:
        return assign_unmatched_face_to_source(1)
    
    # Regular reference face selection
    return append_reference_face(
        src_gallery, dest_gallery, reference_frame_number,
        src_face_idx=1,
        selector_face_index=selector_face_index,
        current_selected_faces=current_selected_faces_2
    )


def remove_reference_face_2(gallery, preview_frame_number):
    return delete_reference_face(
        gallery, preview_frame_number, 1
    )


def assign_unmatched_face_to_source(source_idx: int) -> Tuple[gradio.update, gradio.update, gradio.update, gradio.update]:
    """Assign selected unmatched face to specified source and update face selector"""
    global current_selected_gallery_index, gallery_to_face_mapping, all_unmatched_faces
    
    if current_selected_gallery_index is None:
        return (
            gradio.update(),
            gradio.update(),
            gradio.update(),
            gradio.update()
        )
    
    if current_selected_gallery_index not in gallery_to_face_mapping:
        return (
            gradio.update(),
            gradio.update(),
            gradio.update(),
            gradio.update()
        )
    
    try:
        frame_num, face_idx = gallery_to_face_mapping[current_selected_gallery_index]
        
        # Add to reference face dict (same format as face_selector)
        reference_face_dict = state_manager.get_item('reference_face_dict') or {}
        if source_idx not in reference_face_dict:
            reference_face_dict[source_idx] = []
        
        # Check if this face reference already exists
        ref_face_list = reference_face_dict[source_idx]
        found = any(
            entry["frame_number"] == frame_num and
            entry["face_index"] == face_idx
            for entry in ref_face_list
        )
        
        if not found:
            # Add the face reference
            sorts = current_sort_values()
            face_reference = {
                "frame_number": frame_num,
                "face_index": face_idx,
                "sorts": sorts
            }
            
            reference_face_dict[source_idx].append(face_reference)
            state_manager.set_item('reference_face_dict', reference_face_dict)
            logger.info(f"Added unmatched face reference: frame={frame_num}, face_idx={face_idx}, source={source_idx}", __name__)
            
            # Update reference galleries
            reference_gallery_update = update_reference_position_gallery()
            
            # Update preview image
            from facefusion.uis.components.preview import update_preview_image
            preview_image, enable_button, disable_button = update_preview_image(frame_num)
            
            # Auto-refresh unmatched faces
            updated_gallery, status_msg, ref_gallery, face_input, ignore_btn = find_and_display_all_unmatched_faces()
            
            source_name = "Source 1" if source_idx == 0 else "Source 2"
            
            # Return the appropriate gallery update based on source
            if source_idx == 0:
                return (
                    reference_gallery_update[1],  # REFERENCE_FACES_SELECTION_GALLERY
                    preview_image,
                    enable_button,
                    disable_button
                )
            else:
                return (
                    reference_gallery_update[2],  # REFERENCE_FACES_SELECTION_GALLERY_2
                    preview_image,
                    enable_button,
                    disable_button
                )
        else:
            # Face already exists
            from facefusion.uis.components.preview import update_preview_image
            preview_image, enable_button, disable_button = update_preview_image(frame_num)
            return (
                gradio.update(),
                preview_image,
                enable_button,
                disable_button
            )
        
    except Exception as e:
        logger.error(f"Error assigning unmatched face to source: {e}", __name__)
        return (
            gradio.update(),
            gradio.update(),
            gradio.update(),
            gradio.update()
        )


def reference_frame_back(reference_frame_number: int) -> None:
    frames_per_second = int(detect_video_fps(state_manager.get_item('target_path')))
    reference_frame_number = max(0, reference_frame_number - frames_per_second)
    return update_reference_frame_number_and_gallery(reference_frame_number)


def reference_frame_forward(reference_frame_number: int) -> None:
    frames_per_second = int(detect_video_fps(state_manager.get_item('target_path')))
    reference_frame_number = min(reference_frame_number + frames_per_second, count_video_frame_total(
        state_manager.get_item('target_path')))
    return update_reference_frame_number_and_gallery(reference_frame_number)


def reference_frame_back_five(reference_frame_number: int) -> None:
    frames_per_second = int(detect_video_fps(state_manager.get_item('target_path')))
    reference_frame_number = max(0, reference_frame_number - 5 * frames_per_second)
    return update_reference_frame_number_and_gallery(reference_frame_number)


def reference_frame_forward_five(reference_frame_number: int) -> None:
    frames_per_second = int(detect_video_fps(state_manager.get_item('target_path')))
    reference_frame_number = min(reference_frame_number + 5 * frames_per_second, count_video_frame_total(
        state_manager.get_item('target_path')))
    return update_reference_frame_number_and_gallery(reference_frame_number)


def update_selector_face_index(event_data: SelectData) -> None:
    global selector_face_index
    global selected_face_index
    global selected_face_index_2

    if isinstance(event_data, SelectData):
        selector_face_index = event_data.index  # Extract index
        selected_face_index = -1
        selected_face_index_2 = -1
        print(f"Selector face index updated to: {selector_face_index}")
    else:
        print(f"Unexpected event data: {event_data}")


def update_selected_face_index(event_data: SelectData) -> None:
    global selected_face_index
    global selected_face_index_2
    global selector_face_index
    if isinstance(event_data, SelectData):
        selected_face_index = event_data.index  # Extract index
        selected_face_index_2 = -1
        selector_face_index = -1
        print(f"Selected face index updated to: {selected_face_index}")
    else:
        print(f"Unexpected event data: {event_data}")


def update_selected_face_index_2(event_data: SelectData) -> None:
    global selected_face_index_2
    global selected_face_index
    global selector_face_index
    if isinstance(event_data, SelectData):
        selected_face_index_2 = event_data.index  # Extract index
        selected_face_index = -1
        selector_face_index = -1
        print(f"Selected face index 2 updated to: {selected_face_index_2}")
    else:
        print(f"Unexpected event data: {event_data}")


# Face Cache Functions

def index_video_faces() -> Tuple[gradio.update, gradio.update]:
    """Index all faces in the current video"""
    global indexing_in_progress
    
    with _indexing_lock:
        if indexing_in_progress:
            return (
                gradio.update(value="Indexing already in progress..."),
                gradio.update()
            )
        
        indexing_in_progress = True
    
    try:
        target_path = state_manager.get_item('target_path')
        
        if not target_path or not is_video(target_path):
            return (
                gradio.update(value="❌ No video file selected"),
                gradio.update()
            )
        
        # Set current video path for the index
        VIDEO_FACE_INDEX.current_video_path = target_path
        
        # Check if already indexed
        is_indexed, metadata = VIDEO_FACE_INDEX.is_video_indexed(target_path)
        if is_indexed and metadata.get('indexed_frames', 0) > 0:
            cache_info = f"✅ Already indexed: {metadata['indexed_frames']}/{metadata['total_frames']} frames"
            return (
                gradio.update(value="✅ Video already indexed"),
                gradio.update(value=cache_info)
            )
        
        def progress_callback(progress: float):
            # This would ideally update a progress bar, but gradio limitations...
            pass
        
        # Start indexing
        success = VIDEO_FACE_INDEX.index_video_faces(target_path, progress_callback)
        
        if success:
            # Get updated metadata
            is_indexed, metadata = VIDEO_FACE_INDEX.is_video_indexed(target_path)
            cache_info = f"✅ Indexed: {metadata['indexed_frames']}/{metadata['total_frames']} frames"
            return (
                gradio.update(value="✅ Video indexing completed successfully"),
                gradio.update(value=cache_info)
            )
        else:
            return (
                gradio.update(value="❌ Video indexing failed"),
                gradio.update(value="❌ Indexing failed")
            )
    
    except Exception as e:
        logger.error(f"Error during video indexing: {e}", __name__)
        return (
            gradio.update(value=f"❌ Error: {str(e)}"),
            gradio.update(value="❌ Error occurred")
        )
    
    finally:
        with _indexing_lock:
            indexing_in_progress = False


def find_unmatched_faces() -> Dict[int, List[Face]]:
    """Find faces that don't match reference faces (like face_swapper does)"""
    target_path = state_manager.get_item('target_path')
    
    if not target_path or not is_video(target_path):
        return {}
    
    # Get reference faces from face selector (like face_swapper does)
    reference_faces = {}
    try:
        reference_faces = get_reference_faces()
        if not reference_faces:
            logger.warning("No reference faces available for matching", __name__)
            return {}
    except Exception as e:
        logger.error(f"Error getting reference faces: {e}", __name__)
        return {}
    
    # Get face distance threshold
    face_distance_threshold = state_manager.get_item('reference_face_distance') or 0.6
    
    # Find unmatched faces using video index
    unmatched_faces = VIDEO_FACE_INDEX.find_unmatched_faces(
        target_path, reference_faces, face_distance_threshold
    )
    
    return unmatched_faces


def group_similar_faces(unmatched_faces: Dict[int, List[Face]], similarity_threshold: float = 0.85, min_group_size: int = 10) -> List[Tuple[int, int, Face]]:
    """
    Group similar faces across frames and return representative faces
    Only groups faces that appear in at least min_group_size frames (default: 10 frames = ~0.33 seconds at 30fps)
    Returns: [(frame_number, face_index, face), ...]
    """
    all_faces = []
    face_groups = []
    
    # Collect all faces with their frame/index info, sorted by frame number
    for frame_num in sorted(unmatched_faces.keys()):
        faces = unmatched_faces[frame_num]
        for face_idx, face in enumerate(faces):
            all_faces.append((frame_num, face_idx, face))
    
    logger.info(f"Grouping {len(all_faces)} total unmatched faces with similarity threshold {similarity_threshold}", __name__)
    
    # Group similar faces
    for frame_num, face_idx, face in all_faces:
        # Check if this face belongs to an existing group
        added_to_group = False
        for group in face_groups:
            # Compare with multiple faces in the group for better accuracy
            group_matches = 0
            comparison_faces = min(3, len(group))  # Compare with up to 3 faces from the group
            
            for i in range(comparison_faces):
                representative_face = group[i][2]  # face from tuple
                distance = calc_face_distance(face, representative_face)
                
                if distance < similarity_threshold:
                    group_matches += 1
            
            # Require majority match for grouping (at least 2/3 or 1/1)
            if group_matches >= max(1, comparison_faces // 2):
                group.append((frame_num, face_idx, face))
                added_to_group = True
                break
        
        if not added_to_group:
            # Create new group
            face_groups.append([(frame_num, face_idx, face)])
    
    # Filter groups and return representative faces
    representative_faces = []
    total_grouped = 0
    
    for group in face_groups:
        # Sort group by frame number
        group.sort(key=lambda x: x[0])  # Sort by frame_number
        
        if len(group) >= min_group_size:
            # This is a significant group - use representative
            representative_faces.append(group[0])  # First occurrence
            total_grouped += len(group)
            
            # Calculate frame span for logging
            first_frame = group[0][0]
            last_frame = group[-1][0]
            frame_span = last_frame - first_frame
            
            logger.info(f"Grouped {len(group)} similar faces spanning frames {first_frame}-{last_frame} ({frame_span} frames), representing with frame {first_frame}", __name__)
        else:
            # Small group - include all faces individually
            for face_tuple in group:
                representative_faces.append(face_tuple)
    
    logger.info(f"Face grouping results: {len(representative_faces)} representative faces shown, {total_grouped} faces grouped", __name__)
    return representative_faces


def find_and_display_all_unmatched_faces() -> Tuple[gradio.update, gradio.update, gradio.update, gradio.update, gradio.update]:
    """Find all unmatched faces and display them in the gallery"""
    global all_unmatched_faces, gallery_to_face_mapping, current_selected_gallery_index
    
    try:
        # Clear previous state
        all_unmatched_faces.clear()
        gallery_to_face_mapping.clear()
        current_selected_gallery_index = None
        
        # Find unmatched faces
        unmatched_faces_dict = find_unmatched_faces()
        
        if not unmatched_faces_dict:
            return (
                gradio.update(value=None, visible=False),
                gradio.update(value="✅ No unmatched faces found"),
                gradio.update(visible=True),  # Show reference gallery
                gradio.update(visible=False),  # Hide face number input
                gradio.update(visible=False)   # Hide ignore button
            )
        
        # Group similar faces to reduce clutter
        # At 30fps: 10 frames = 0.33s, 30 frames = 1s, 60 frames = 2s
        min_group_frames = 15  # Minimum frames to consider grouping (0.5 seconds at 30fps)
        similarity_threshold = 0.85  # Higher threshold for more strict grouping
        representative_faces = group_similar_faces(unmatched_faces_dict, similarity_threshold, min_group_frames)
        
        # Get target video path
        target_path = state_manager.get_item('target_path')
        if not target_path:
            return (
                gradio.update(value=None, visible=False),
                gradio.update(value="❌ No video file selected"),
                gradio.update(visible=True),
                gradio.update(visible=False),
                gradio.update(visible=False)
            )
        
        # Extract face crops for gallery
        gallery_frames = []
        
        for gallery_idx, (frame_num, face_idx, face) in enumerate(representative_faces):
            # Store the mapping
            all_unmatched_faces.append((frame_num, face_idx, face))
            gallery_to_face_mapping[gallery_idx] = (frame_num, face_idx)
            
            # Get the vision frame
            vision_frame = get_video_frame(target_path, frame_num)
            if vision_frame is None:
                continue
            
            # Extract face crop
            start_x, start_y, end_x, end_y = map(int, face.bounding_box)
            padding_x = int((end_x - start_x) * 0.25)
            padding_y = int((end_y - start_y) * 0.25)
            start_x = max(0, start_x - padding_x)
            start_y = max(0, start_y - padding_y)
            end_x = max(0, end_x + padding_x)
            end_y = max(0, end_y + padding_y)
            
            crop_vision_frame = vision_frame[start_y:end_y, start_x:end_x]
            crop_vision_frame = normalize_frame_color(crop_vision_frame)
            gallery_frames.append(crop_vision_frame)
            
            logger.debug(f"Added face from frame {frame_num}, face {face_idx} to gallery index {gallery_idx}", __name__)
        
        total_faces = sum(len(faces) for faces in unmatched_faces_dict.values())
        logger.info(f"Found {total_faces} total unmatched faces, showing {len(representative_faces)} representative faces", __name__)
        
        return (
            gradio.update(value=gallery_frames, visible=True),
            gradio.update(value=f"✅ Found {len(representative_faces)} unmatched faces ({total_faces} total)"),
            gradio.update(visible=False),  # Hide reference gallery when showing unmatched
            gradio.update(visible=True),   # Show face number input
            gradio.update(visible=True)    # Show ignore button
        )
        
    except Exception as e:
        logger.error(f"Error finding unmatched faces: {e}", __name__)
        return (
            gradio.update(value=None, visible=False),
            gradio.update(value=f"❌ Error: {str(e)}"),
            gradio.update(visible=True),
            gradio.update(visible=False),
            gradio.update(visible=False)
        )


def select_unmatched_face_and_jump(event_data: SelectData) -> Tuple[gradio.update, gradio.update, gradio.update, gradio.update]:
    """Handle selection of an unmatched face and jump to that frame"""
    global current_selected_gallery_index, gallery_to_face_mapping
    
    logger.debug(f"select_unmatched_face_and_jump called with event_data: {event_data}, type: {type(event_data)}", __name__)
    
    if not isinstance(event_data, SelectData):
        logger.warning(f"Invalid event data type: {type(event_data)}", __name__)
        return (
            gradio.update(value="❌ Invalid selection event"),
            gradio.update(),
            gradio.update(),
            gradio.update()
        )
    
    gallery_index = event_data.index
    current_selected_gallery_index = gallery_index
    
    if gallery_index not in gallery_to_face_mapping:
        logger.warning(f"Gallery index {gallery_index} not found in mapping. Available indices: {list(gallery_to_face_mapping.keys())}", __name__)
        return (
            gradio.update(value="❌ Face selection mapping error"),
            gradio.update(),
            gradio.update(),
            gradio.update()
        )
    
    try:
        frame_num, face_idx = gallery_to_face_mapping[gallery_index]
        
        # Jump to the frame
        state_manager.set_item('reference_frame_number', frame_num)
        
        # Update preview image
        from facefusion.uis.components.preview import update_preview_image
        preview_image, enable_button, disable_button = update_preview_image(frame_num)
        
        # Update reference position gallery for the new frame
        reference_gallery_update = update_reference_position_gallery()
        
        logger.info(f"Selected gallery index {gallery_index} -> frame {frame_num}, face {face_idx}, jumped to frame", __name__)
        
        return (
            gradio.update(value=f"✅ Selected face from frame {frame_num} (gallery #{gallery_index + 1}) - Jumped to frame"),
            gradio.update(value=frame_num),
            preview_image,
            reference_gallery_update[0]  # Just the gallery update
        )
        
    except Exception as e:
        logger.error(f"Error selecting unmatched face and jumping: {e}", __name__)
        return (
            gradio.update(value=f"❌ Error: {str(e)}"),
            gradio.update(),
            gradio.update(),
            gradio.update()
        )


def select_face_by_number(face_number: Optional[float]) -> gradio.update:
    """Select face by number input (1-based indexing)"""
    global current_selected_gallery_index, gallery_to_face_mapping
    
    if face_number is None:
        return gradio.update(value="")
    
    try:
        # Convert to 0-based index
        gallery_index = int(face_number) - 1
        
        if gallery_index < 0:
            return gradio.update(value="❌ Face number must be 1 or higher")
        
        if gallery_index in gallery_to_face_mapping:
            current_selected_gallery_index = gallery_index
            frame_num, face_idx = gallery_to_face_mapping[gallery_index]
            logger.info(f"Selected face by number: {face_number} -> gallery index {gallery_index} -> frame {frame_num}, face {face_idx}", __name__)
            return gradio.update(value=f"✅ Selected face #{int(face_number)} from frame {frame_num}")
        else:
            max_face_num = len(gallery_to_face_mapping)
            return gradio.update(value=f"❌ Face number {int(face_number)} not found. Available: 1-{max_face_num}")
            
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting face number to int: {e}, face_number: {face_number}", __name__)
        return gradio.update(value="❌ Invalid face number")


def ignore_selected_face() -> Tuple[gradio.update, gradio.update]:
    """Mark selected face as ignored and auto-refresh"""
    global current_selected_gallery_index, gallery_to_face_mapping
    
    if current_selected_gallery_index is None:
        return (
            gradio.update(value="❌ Please select a face first"),
            gradio.update()
        )
    
    if current_selected_gallery_index not in gallery_to_face_mapping:
        return (
            gradio.update(value="❌ Invalid face selection"),
            gradio.update()
        )
    
    try:
        frame_num, face_idx = gallery_to_face_mapping[current_selected_gallery_index]
        
        # Get the face object
        selected_face = None
        for stored_frame, stored_face_idx, face in all_unmatched_faces:
            if stored_frame == frame_num and stored_face_idx == face_idx:
                selected_face = face
                break
        
        if selected_face is None:
            return (
                gradio.update(value="❌ Could not find selected face"),
                gradio.update()
            )
        
        # Mark face as ignored in video index
        target_path = state_manager.get_item('target_path')
        VIDEO_FACE_INDEX.mark_face_as_ignored(target_path, frame_num, face_idx, selected_face)
        
        # Auto-refresh unmatched faces
        updated_gallery, status_msg, ref_gallery, face_input, ignore_btn = find_and_display_all_unmatched_faces()
        
        return (
            gradio.update(value="✅ Face ignored forever - Auto-refreshed unmatched faces"),
            updated_gallery
        )
        
    except Exception as e:
        logger.error(f"Error ignoring face: {e}", __name__)
        return (
            gradio.update(value=f"❌ Error: {str(e)}"),
            gradio.update()
        )


def clear_video_cache() -> Tuple[gradio.update, gradio.update, gradio.update]:
    """Clear the video face cache"""
    global all_unmatched_faces, gallery_to_face_mapping, current_selected_gallery_index
    
    try:
        target_path = state_manager.get_item('target_path')
        
        # Clear our tracking
        all_unmatched_faces.clear()
        gallery_to_face_mapping.clear()
        current_selected_gallery_index = None
        
        if target_path:
            VIDEO_FACE_INDEX.clear_video_index(target_path)
            return (
                gradio.update(value="✅ Video cache cleared"),
                gradio.update(value="Cache cleared"),
                gradio.update(value=None, visible=False)
            )
        else:
            VIDEO_FACE_INDEX.clear_video_index()  # Clear all
            return (
                gradio.update(value="✅ All caches cleared"),
                gradio.update(value="All caches cleared"),
                gradio.update(value=None, visible=False)
            )
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", __name__)
        return (
            gradio.update(value=f"❌ Error: {str(e)}"),
            gradio.update(value="Error clearing cache"),
            gradio.update()
        )


def check_and_update_cache_status() -> Tuple[gradio.update, gradio.update]:
    """Check if cache exists for current video and update UI status"""
    try:
        target_path = state_manager.get_item('target_path')
        
        if not target_path or not is_video(target_path):
            return (
                gradio.update(value="No video selected"),
                gradio.update(value="Select a video to see cache info")
            )
        
        # Check if cache exists
        cache_exists = VIDEO_FACE_INDEX.has_video_index(target_path)
        
        if cache_exists:
            # Get cache info
            cache_info = VIDEO_FACE_INDEX.get_video_cache_info(target_path)
            total_frames = cache_info.get('total_frames', 0)
            total_faces = cache_info.get('total_faces', 0)
            cache_date = cache_info.get('created_at', 'Unknown')
            
            status_msg = f"✅ Cache found for current video"
            info_msg = f"Frames: {total_frames}, Faces: {total_faces}, Created: {cache_date}"
        else:
            status_msg = "No cache found for current video"
            info_msg = "Click 'Index Video Faces' to create cache"
        
        return (
            gradio.update(value=status_msg),
            gradio.update(value=info_msg)
        )
        
    except Exception as e:
        logger.error(f"Error checking cache status: {e}", __name__)
        return (
            gradio.update(value=f"❌ Error checking cache: {str(e)}"),
            gradio.update(value="Error getting cache info")
        )
