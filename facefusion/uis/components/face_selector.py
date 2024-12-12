from typing import List, Optional, Tuple

import gradio
import numpy as np
from gradio import SelectData

import facefusion.choices
from facefusion import state_manager, wording
from facefusion.common_helper import calc_float_step, calc_int_step
from facefusion.face_analyser import get_many_faces
from facefusion.face_selector import sort_and_filter_faces
from facefusion.face_store import clear_reference_faces, clear_static_faces
from facefusion.filesystem import is_image, is_video
from facefusion.typing import FaceSelectorMode, VisionFrame, Race, Gender, FaceSelectorOrder
from facefusion.uis.components.face_masker import clear_mask_times
from facefusion.uis.core import get_ui_component, register_ui_component, get_ui_components
# from gradio_rangeslider import RangeSlider
from facefusion.uis.typing import ComponentOptions
from facefusion.uis.ui_helper import convert_str_none
from facefusion.vision import get_video_frame, normalize_frame_color, read_static_image, detect_video_fps, \
    count_video_frame_total

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
    non_face_processors = ['frame_colorizer', 'frame_enhancer']
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
            toggle_selector_group,
            inputs=processors_checkbox_group,
            outputs=[FACE_SELECTOR_GROUP]
        )


def toggle_selector_group(processors: List[str]) -> gradio.update:
    non_face_processors = ['frame_colorizer', 'frame_enhancer']
    # Make the group visible if any face processor is selected
    for processor in processors:
        if processor not in non_face_processors:
            return gradio.update(visible=True)
    return gradio.update(visible=False)


def update_face_selector_mode(face_selector_mode: FaceSelectorMode) -> Tuple[gradio.Gallery, gradio.update]:
    state_manager.set_item('face_selector_mode', face_selector_mode)
    if face_selector_mode == 'many':
        return gradio.update(visible=False), gradio.update(visible=False)
    if face_selector_mode == 'one':
        return gradio.update(visible=False), gradio.update(visible=False)
    if face_selector_mode == 'reference':
        return gradio.update(visible=True), gradio.update(visible=True)


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


def add_reference_face(src_gallery, dest_gallery, reference_frame_number) -> gradio.Gallery:
    global selected_face_index
    global current_selected_faces
    global selector_face_index
    dest_items = [item["name"] for item in dest_gallery]
    if src_gallery is not None and any(src_gallery):
        # If the number of items in gallery is less than selected_face_index, then the selected_face_index is invalid
        if len(src_gallery) <= selector_face_index:
            selector_face_index = -1
            print("Invalid index")
            return gradio.update()
        selected_item = src_gallery[selector_face_index]
        face_data = current_reference_faces[selector_face_index]
        reference_face_dict = state_manager.get_item('reference_face_dict')
        if not reference_face_dict:
            reference_face_dict = {}
        if reference_frame_number not in reference_face_dict:
            reference_face_dict[reference_frame_number] = []
            state_manager.set_item('reference_face_dict', reference_face_dict)
        found = False
        for existing_face_data in state_manager.get_item('reference_face_dict')[reference_frame_number]:
            if np.array_equal(face_data, existing_face_data):
                found = True
                break

        if not found:
            if not reference_face_dict:
                reference_face_dict = {}
            reference_face_dict[reference_frame_number].append(face_data)
            state_manager.set_item('reference_face_dict', reference_face_dict)
            current_selected_faces.append(face_data)
            dest_items.append(selected_item["name"])
        from facefusion.uis.components.preview import update_preview_image

        preview, enable_button, disable_button = update_preview_image(reference_frame_number)
        return gradio.update(value=dest_items), preview, enable_button, disable_button
    else:
        return gradio.update(), gradio.update(), gradio.update(), gradio.update()  # Return the original gallery if no item is selected or if it's empty


def remove_reference_face(gallery: gradio.Gallery, preview_frame_number) -> gradio.Gallery:
    global selected_face_index, current_selected_faces
    if len(gallery) <= selected_face_index or len(current_selected_faces) <= selected_face_index:
        selected_face_index = -1
        print("Invalid index")
        return gradio.update()

    # Remove the selected item from the gallery
    new_items = []
    gallery_index = 0
    for item in gallery:
        if gallery_index != selected_face_index:
            new_items.append(item["name"])
        gallery_index += 1
    global_reference_faces = state_manager.get_item('reference_face_dict')
    if not global_reference_faces:
        global_reference_faces = {}
    face_to_remove = current_selected_faces[selected_face_index]
    found = False
    for frame_no, faces in global_reference_faces.items():
        cleaned_faces = []
        for existing_face_data in faces:
            if np.array_equal(face_to_remove, existing_face_data):
                print("Found face to remove")
                found = True
                continue
            cleaned_faces.append(existing_face_data)
        global_reference_faces[frame_no] = cleaned_faces
        if found:
            break

    state_manager.set_item('reference_face_dict', global_reference_faces)
    current_selected_faces.pop(selected_face_index)
    from facefusion.uis.components.preview import update_preview_image
    preview_image, enable_button, disable_button = update_preview_image(preview_frame_number)
    return gradio.update(value=new_items), preview_image, enable_button, disable_button


def add_reference_face_2(src_gallery, dest_gallery, reference_frame_number) -> gradio.Gallery:
    global selector_face_index, current_selected_faces_2
    dest_items = [item["name"] for item in dest_gallery]
    if src_gallery is not None and any(src_gallery):
        # If the number of items in gallery is less than selected_face_index, then the selected_face_index is invalid
        if len(src_gallery) <= selector_face_index:
            selector_face_index = -1
            print("Invalid index")
            return gradio.update()
        selected_item = src_gallery[selector_face_index]
        face_data = current_reference_faces[selector_face_index]
        reference_face_dict = state_manager.get_item('reference_face_dict_2')
        if not reference_face_dict:
            reference_face_dict = {}
        if reference_frame_number not in reference_face_dict:
            reference_face_dict[reference_frame_number] = []
            state_manager.set_item('reference_face_dict_2', reference_face_dict)
        found = False
        for existing_face_data in state_manager.get_item('reference_face_dict_2')[reference_frame_number]:
            if np.array_equal(face_data, existing_face_data):
                found = True
                break

        if not found:
            reference_face_dict[reference_frame_number].append(face_data)
            state_manager.set_item('reference_face_dict_2', reference_face_dict)
            current_selected_faces_2.append(face_data)
            dest_items.append(selected_item["name"])
        from facefusion.uis.components.preview import update_preview_image

        preview, enable_button, disable_button = update_preview_image(reference_frame_number)
        return gradio.update(value=dest_items), preview, enable_button, disable_button
    else:
        return gradio.update(), gradio.update(), gradio.update(), gradio.update()  # Return the original gallery if no item is selected or if it's empty


def remove_reference_face_2(gallery: gradio.Gallery, preview_frame_number) -> gradio.Gallery:
    global selected_face_index_2, current_selected_faces_2
    if len(gallery) <= selected_face_index_2 or len(current_selected_faces) <= selected_face_index_2:
        selected_face_index_2 = -1
        print("Invalid index")
        return gradio.update()

    # Remove the selected item from the gallery
    new_items = []
    gallery_index = 0
    for item in gallery:
        if gallery_index != selected_face_index_2:
            new_items.append(item["name"])
        gallery_index += 1
    global_reference_faces = state_manager.get_item('reference_face_dict_2')
    if not global_reference_faces:
        global_reference_faces = {}
    face_to_remove = current_selected_faces[selected_face_index_2]
    found = False
    for frame_no, faces in global_reference_faces.items():
        cleaned_faces = []
        for existing_face_data in faces:
            if np.array_equal(face_to_remove, existing_face_data):
                print("Found face to remove")
                found = True
                continue
            cleaned_faces.append(existing_face_data)
        global_reference_faces[frame_no] = cleaned_faces
        if found:
            break

    state_manager.set_item('reference_face_dict_2', global_reference_faces)
    current_selected_faces_2.pop(selected_face_index_2)
    from facefusion.uis.components.preview import update_preview_image
    preview, enable_button, disable_button = update_preview_image(preview_frame_number)
    return gradio.update(value=new_items), preview, enable_button, disable_button


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
