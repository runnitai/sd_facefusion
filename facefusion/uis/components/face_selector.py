from typing import List, Optional, Tuple, Any, Dict

import gradio
import numpy as np

import facefusion.globals
import facefusion.choices
from facefusion import wording
from facefusion.face_store import clear_static_faces, clear_reference_faces
from facefusion.vision import get_video_frame, read_static_image, normalize_frame_color, count_video_frame_total, \
    detect_fps
from facefusion.face_analyser import get_many_faces
from facefusion.typing import Frame, FaceSelectorMode, Face
from facefusion.filesystem import is_image, is_video
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.uis.typing import ComponentName

FACE_SELECTOR_MODE_DROPDOWN: Optional[gradio.Dropdown] = None
REFERENCE_FACE_POSITION_GALLERY: Optional[gradio.Gallery] = None
REFERENCE_FACE_DISTANCE_SLIDER: Optional[gradio.Slider] = None
ADD_REFERENCE_FACE_BUTTON: Optional[gradio.Button] = None
REMOVE_REFERENCE_FACE_BUTTON: Optional[gradio.Button] = None
REFERENCE_FACES_SELECTION_GALLERY: Optional[gradio.Gallery] = None

# Stores face data for current reference frame
current_reference_faces = []
# Stores the actual images
current_reference_frames = []
# Stores the selected reference face data
current_selected_faces = []
# The selected face in the gallery
selected_face_index = -1


def render() -> None:
    global FACE_SELECTOR_MODE_DROPDOWN
    global REFERENCE_FACE_POSITION_GALLERY
    global REFERENCE_FACE_DISTANCE_SLIDER
    global REFERENCE_FACES_SELECTION_GALLERY
    global ADD_REFERENCE_FACE_BUTTON
    global REMOVE_REFERENCE_FACE_BUTTON

    reference_face_gallery_args: Dict[str, Any] = \
        {
            'label': wording.get('reference_face_gallery_label'),
            'object_fit': 'cover',
            'columns': 8,
            'allow_preview': False,
            'visible': 'reference' in facefusion.globals.face_selector_mode
        }
    if is_image(facefusion.globals.target_path):
        reference_frame = read_static_image(facefusion.globals.target_path)
        reference_face_gallery_args['value'] = extract_gallery_frames(reference_frame)
    if is_video(facefusion.globals.target_path):
        reference_frame = get_video_frame(facefusion.globals.target_path, facefusion.globals.reference_frame_number)
        reference_face_gallery_args['value'] = extract_gallery_frames(reference_frame)
    FACE_SELECTOR_MODE_DROPDOWN = gradio.Dropdown(
        label=wording.get('face_selector_mode_dropdown_label'),
        choices=facefusion.choices.face_selector_modes,
        value=facefusion.globals.face_selector_mode,
        elem_id='ff_face_recognition_dropdown',
    )
    with gradio.Row():
        ADD_REFERENCE_FACE_BUTTON = gradio.Button(
            value="+",
            elem_id='ff_add_reference_face_button'
        )
        REFERENCE_FACE_POSITION_GALLERY = gradio.Gallery(**reference_face_gallery_args,
                                                         elem_id='ff_reference_face_position_gallery')
    with gradio.Row():
        REMOVE_REFERENCE_FACE_BUTTON = gradio.Button(
            value="-",
            variant='secondary',
            elem_id='ff_remove_reference_faces_button'
        )
        REFERENCE_FACES_SELECTION_GALLERY = gradio.Gallery(
            label="Selected Reference Faces",
            object_fit='cover',
            columns=8,
            allow_preview=False,
            visible='reference' in facefusion.globals.face_selector_mode,
            elem_id='ff_reference_faces_selection_gallery'
        )
    REFERENCE_FACE_DISTANCE_SLIDER = gradio.Slider(
        label=wording.get('reference_face_distance_slider_label'),
        value=facefusion.globals.reference_face_distance,
        step=facefusion.choices.reference_face_distance_range[1] - facefusion.choices.reference_face_distance_range[0],
        minimum=facefusion.choices.reference_face_distance_range[0],
        maximum=facefusion.choices.reference_face_distance_range[-1],
        visible='reference' in facefusion.globals.face_selector_mode,
        elem_id='ff_reference_face_distance_slider'
    )
    register_ui_component('face_selector_mode_dropdown', FACE_SELECTOR_MODE_DROPDOWN)
    register_ui_component('reference_face_position_gallery', REFERENCE_FACE_POSITION_GALLERY)
    register_ui_component('reference_faces_selection_gallery', REFERENCE_FACES_SELECTION_GALLERY)
    register_ui_component('reference_face_distance_slider', REFERENCE_FACE_DISTANCE_SLIDER)
    register_ui_component('add_reference_face_button', ADD_REFERENCE_FACE_BUTTON)
    register_ui_component('remove_reference_faces_button', REMOVE_REFERENCE_FACE_BUTTON)


def listen() -> None:
    galleries = [REFERENCE_FACE_POSITION_GALLERY, REFERENCE_FACES_SELECTION_GALLERY]
    FACE_SELECTOR_MODE_DROPDOWN.select(update_face_selector_mode, inputs=FACE_SELECTOR_MODE_DROPDOWN,
                                       outputs=[REFERENCE_FACE_POSITION_GALLERY, REFERENCE_FACES_SELECTION_GALLERY,
                                                REFERENCE_FACE_DISTANCE_SLIDER, ADD_REFERENCE_FACE_BUTTON,
                                                REMOVE_REFERENCE_FACE_BUTTON])
    REFERENCE_FACE_POSITION_GALLERY.select(update_selected_face_index)
    REFERENCE_FACES_SELECTION_GALLERY.select(update_selected_face_index)
    REFERENCE_FACE_DISTANCE_SLIDER.change(update_reference_face_distance, inputs=REFERENCE_FACE_DISTANCE_SLIDER)
    multi_component_names: List[ComponentName] = \
        [
            'target_image',
            'target_video'
        ]
    for component_name in multi_component_names:
        component = get_ui_component(component_name)
        if component:
            for method in ['upload', 'change', 'clear']:
                getattr(component, method)(update_reference_face_position)
                getattr(component, method)(update_reference_position_gallery, outputs=galleries)
    change_one_component_names: List[ComponentName] = \
        [
            'face_analyser_order_dropdown',
            'face_analyser_age_dropdown',
            'face_analyser_gender_dropdown'
        ]
    for component_name in change_one_component_names:
        component = get_ui_component(component_name)
        if component:
            component.change(update_reference_position_gallery, outputs=galleries)
    change_two_component_names: List[ComponentName] = \
        [
            'face_detector_model_dropdown',
            'face_detector_size_dropdown',
            'face_detector_score_slider'
        ]
    for component_name in change_two_component_names:
        component = get_ui_component(component_name)
        if component:
            component.change(clear_and_update_reference_position_gallery, outputs=galleries)
    preview_frame_slider = get_ui_component('preview_frame_slider')
    preview_frame_back_button = get_ui_component('preview_frame_back_button')
    preview_frame_forward_button = get_ui_component('preview_frame_forward_button')
    preview_image = get_ui_component('preview_image')
    if preview_frame_slider:
        # update_preview_image, inputs=PREVIEW_FRAME_SLIDER, outputs=PREVIEW_IMAGE
        ADD_REFERENCE_FACE_BUTTON.click(add_reference_face,
                                        inputs=[REFERENCE_FACE_POSITION_GALLERY, REFERENCE_FACES_SELECTION_GALLERY,
                                                preview_frame_slider],
                                        outputs=[REFERENCE_FACES_SELECTION_GALLERY, preview_image])
        REMOVE_REFERENCE_FACE_BUTTON.click(fn=remove_reference_face,
                                           inputs=[REFERENCE_FACES_SELECTION_GALLERY, preview_frame_slider],
                                           outputs=[REFERENCE_FACES_SELECTION_GALLERY, preview_image])

        preview_frame_back_button.click(reference_frame_back,
                                        inputs=preview_frame_slider, outputs=[preview_frame_slider,
                                                                              REFERENCE_FACE_POSITION_GALLERY, REFERENCE_FACES_SELECTION_GALLERY])
        preview_frame_forward_button.click(reference_frame_forward, inputs=preview_frame_slider,
                                             outputs=[preview_frame_slider, REFERENCE_FACE_POSITION_GALLERY, REFERENCE_FACES_SELECTION_GALLERY])
        preview_frame_slider.change(update_reference_frame_number_and_gallery, inputs=preview_frame_slider,
                                    outputs=[preview_frame_slider, REFERENCE_FACE_POSITION_GALLERY, REFERENCE_FACES_SELECTION_GALLERY])


def update_face_selector_mode(face_selector_mode: FaceSelectorMode) -> Tuple[
    gradio.update, gradio.update, gradio.update, gradio.update, gradio.update]:
    facefusion.globals.face_selector_mode = face_selector_mode
    visible = 'reference' in face_selector_mode
    return gradio.update(visible=visible), gradio.update(visible=visible), gradio.update(
        visible=visible), gradio.update(visible=visible), gradio.update(visible=visible)


def update_selected_face_index(event: gradio.SelectData) -> None:
    global selected_face_index
    print("Index changed...")
    selected_face_index = event.index


def clear_and_update_reference_face_position(event: gradio.SelectData) -> gradio.Gallery:
    clear_reference_faces()
    clear_static_faces()
    update_reference_face_position(event.index)
    return update_reference_position_gallery()


def add_reference_face(src_gallery, dest_gallery, reference_frame_number) -> gradio.Gallery:
    global selected_face_index
    dest_items = [item["name"] for item in dest_gallery]
    if src_gallery is not None and any(src_gallery):
        # If the number of items in gallery is less than selected_face_index, then the selected_face_index is invalid
        if len(src_gallery) <= selected_face_index:
            selected_face_index = -1
            print("Invalid index")
            return gradio.update()
        selected_item = src_gallery[selected_face_index]
        face_data = current_reference_faces[selected_face_index]
        if reference_frame_number not in facefusion.globals.reference_face_dict:
            facefusion.globals.reference_face_dict[reference_frame_number] = []
        found = False
        for existing_face_data in facefusion.globals.reference_face_dict[reference_frame_number]:
            if np.array_equal(face_data, existing_face_data):
                found = True
                break

        if not found:
            facefusion.globals.reference_face_dict[reference_frame_number].append(face_data)
            current_selected_faces.append(face_data)
            dest_items.append(selected_item["name"])
        from facefusion.uis.components.preview import update_preview_image
        out_preview = update_preview_image(reference_frame_number)
        return gradio.update(value=dest_items), out_preview
    else:
        return gradio.update(), gradio.update()  # Return the original gallery if no item is selected or if it's empty


def remove_reference_face(gallery: gradio.Gallery, preview_frame_number) -> gradio.Gallery:
    global selected_face_index
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
    global_reference_faces = facefusion.globals.reference_face_dict
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

    facefusion.globals.reference_face_dict = global_reference_faces
    current_selected_faces.pop(selected_face_index)
    from facefusion.uis.components.preview import update_preview_image
    preview_image = update_preview_image(preview_frame_number)
    return gradio.update(value=new_items), preview_image


def update_reference_face_position(reference_face_position: int = 0) -> None:
    facefusion.globals.reference_face_position = reference_face_position


def update_reference_face_distance(reference_face_distance: float) -> None:
    facefusion.globals.reference_face_distance = reference_face_distance


def update_reference_frame_number(reference_frame_number: int) -> None:
    facefusion.globals.reference_frame_number = reference_frame_number


def reference_frame_back(reference_frame_number: int) -> None:
    frames_per_second = int(detect_fps(facefusion.globals.target_path))
    reference_frame_number = max(0, reference_frame_number - frames_per_second)
    return update_reference_frame_number_and_gallery(reference_frame_number)


def reference_frame_forward(reference_frame_number: int) -> None:
    frames_per_second = int(detect_fps(facefusion.globals.target_path))
    reference_frame_number = min(reference_frame_number + frames_per_second, count_video_frame_total(
        facefusion.globals.target_path))
    return update_reference_frame_number_and_gallery(reference_frame_number)


def clear_and_update_reference_position_gallery() -> gradio.update:
    clear_reference_faces()
    clear_static_faces()
    return update_reference_position_gallery()


def update_reference_position_gallery() -> Tuple[gradio.update, gradio.update]:
    gallery_frames = []
    selection_gallery = gradio.update()
    if is_image(facefusion.globals.target_path):
        reference_frame = read_static_image(facefusion.globals.target_path)
        gallery_frames = extract_gallery_frames(reference_frame)
    elif is_video(facefusion.globals.target_path):
        reference_frame = get_video_frame(facefusion.globals.target_path, facefusion.globals.reference_frame_number)
        gallery_frames = extract_gallery_frames(reference_frame)
    else:
        selection_gallery = gradio.update(value=None)
        facefusion.globals.reference_face_dict = {}
        global current_selected_faces
        current_selected_faces = []
    if gallery_frames:
        return gradio.update(value=gallery_frames), selection_gallery
    return gradio.update(value=None), selection_gallery


def update_reference_frame_number_and_gallery(reference_frame_number) -> Tuple[gradio.update, gradio.update]:
    gallery_frames = []
    facefusion.globals.reference_frame_number = reference_frame_number
    selection_gallery = gradio.update()
    if is_image(facefusion.globals.target_path):
        reference_frame = read_static_image(facefusion.globals.target_path)
        gallery_frames = extract_gallery_frames(reference_frame)
    elif is_video(facefusion.globals.target_path):
        reference_frame = get_video_frame(facefusion.globals.target_path, facefusion.globals.reference_frame_number)
        gallery_frames = extract_gallery_frames(reference_frame)
    else:
        selection_gallery = gradio.update(value=None)
        facefusion.globals.reference_face_dict = {}
        global current_selected_faces
        current_selected_faces = []
    if gallery_frames:
        return gradio.update(value=reference_frame_number), gradio.update(value=gallery_frames), selection_gallery
    return gradio.update(value=reference_frame_number), gradio.update(value=None), selection_gallery


def extract_gallery_frames(reference_frame: Frame) -> List[Frame]:
    crop_frames = []
    faces = get_many_faces(reference_frame)
    global current_reference_faces
    global current_reference_frames
    current_reference_faces = faces
    for face in faces:
        start_x, start_y, end_x, end_y = map(int, face.bbox)
        padding_x = int((end_x - start_x) * 0.25)
        padding_y = int((end_y - start_y) * 0.25)
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = max(0, end_x + padding_x)
        end_y = max(0, end_y + padding_y)
        crop_frame = reference_frame[start_y:end_y, start_x:end_x]
        crop_frame = normalize_frame_color(crop_frame)
        crop_frames.append(crop_frame)
    current_reference_frames = crop_frames
    return crop_frames
