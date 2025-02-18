import hashlib
from typing import Optional, List, Any, Dict

import numpy

from facefusion import state_manager
from facefusion.face_selector import sort_and_filter_faces
from facefusion.filesystem import is_image, is_video
from facefusion.typing import VisionFrame, Face
from facefusion.vision import read_static_image, get_video_frame


class FaceStore:
    def __init__(self):
        self.store = {}

    def get(self, key: str, default: Any = None) -> Optional[Dict[int, List[Face]]]:
        return self.store.get(key, default)

    def set(self, key: str, faces: Dict[int, List[Face]]) -> None:
        existing_faces = self.store.get(key, {})
        for idx, new_faces in faces.items():
            if idx not in existing_faces:
                existing_faces[idx] = []
            for face in new_faces:
                if not any(numpy.array_equal(face.embedding, existing_face.embedding)
                           for existing_face in existing_faces[idx]):
                    existing_faces[idx].append(face)
        self.store[key] = existing_faces

    def clear(self) -> None:
        self.store = {}


FACE_STORE = FaceStore()


def create_frame_hash(vision_frame: VisionFrame) -> Optional[str]:
    try:
        if not numpy.any(vision_frame):
            return None
        return hashlib.sha1(vision_frame.tobytes()).hexdigest()
    except Exception as e:
        print(f"Error creating frame hash: {e}")
        return None


def get_static_faces(vision_frame: VisionFrame) -> Optional[List[Face]]:
    frame_hash = create_frame_hash(vision_frame)
    if not frame_hash:
        return None

    stored_value = FACE_STORE.get(frame_hash)
    if not stored_value:
        return None

    # Flatten all faces from the dictionary into a single list
    all_faces = []
    for _, faces_ in stored_value.items():
        all_faces.extend(faces_)
    return all_faces


def set_static_faces(vision_frame: VisionFrame, faces: List[Face]) -> None:
    frame_hash = create_frame_hash(vision_frame)
    if frame_hash:
        # Wrap the list of faces into a dict to satisfy FaceStore.set(...)
        FACE_STORE.set(frame_hash, {0: faces})


def clear_static_faces() -> None:
    FACE_STORE.clear()


def get_reference_face(frame_number: int, face_idx: int, sorts: Dict[str, Any]) -> Optional[Face]:
    from facefusion.face_analyser import get_many_faces
    target_path = state_manager.get_item('target_path')
    if is_image(target_path):
        temp_vision_frame = read_static_image(target_path)
    elif is_video(target_path):
        temp_vision_frame = get_video_frame(target_path, frame_number)
    else:
        return None

    sorted_faces = sort_and_filter_faces(get_many_faces([temp_vision_frame]), sorts)
    if sorted_faces and face_idx < len(sorted_faces):
        return sorted_faces[face_idx]
    return None


def get_reference_faces() -> Dict[int, List[Face]]:
    from facefusion.face_analyser import get_average_faces
    global FACE_STORE
    reference_face_dict = state_manager.get_item('reference_face_dict')
    stored_faces = FACE_STORE.get('reference_faces', {})

    for source_face_index, src_face_refs in reference_face_dict.items():
        faces = []
        for face_ref in src_face_refs:
            frame_number = face_ref.get('frame_number')
            face_index = face_ref.get('face_index')
            sorts = face_ref.get('sorts', {})
            ref_faces = stored_faces.get(source_face_index, [])

            # Check if reference face already exists
            existing_face = get_reference_face(frame_number, face_index, sorts)
            if existing_face and not any(
                numpy.array_equal(existing_face.embedding, f.embedding) for f in ref_faces
            ):
                faces.append(existing_face)

        if faces:
            if source_face_index not in stored_faces:
                stored_faces[source_face_index] = []
            stored_faces[source_face_index].extend(faces)

        # Add average faces for the current source_face_index
        source_face_dict = get_average_faces()
        source_face = source_face_dict.get(source_face_index)
        if source_face:
            stored_faces[source_face_index].append(source_face)

    FACE_STORE.set('reference_faces', stored_faces)
    return stored_faces


def clear_reference_faces() -> None:
    FACE_STORE.set('reference_faces', {})
