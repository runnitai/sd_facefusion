import hashlib
from typing import Optional, List, Tuple

import numpy

from facefusion import state_manager
from facefusion.typing import VisionFrame, Face, FaceStore, FaceSet

FACE_STORE: FaceStore = \
    {
        'static_faces': {},
        'reference_faces': {}
    }

FACE_STORE_2: FaceStore = \
    {
        'static_faces': {},
        'reference_faces': {}
    }


def get_face_store() -> FaceStore:
    return FACE_STORE


def get_static_faces(vision_frame: VisionFrame, dict_2=False) -> Optional[List[Face]]:
    frame_hash = create_frame_hash(vision_frame)
    if dict_2:
        if frame_hash in FACE_STORE_2['static_faces']:
            return FACE_STORE_2['static_faces'][frame_hash]
        return None
    if frame_hash in FACE_STORE['static_faces']:
        return FACE_STORE['static_faces'][frame_hash]
    return None


def set_static_faces(vision_frame: VisionFrame, faces: List[Face], dict_2=False) -> None:
    frame_hash = create_frame_hash(vision_frame)
    if frame_hash:
        if dict_2:
            FACE_STORE_2['static_faces'][frame_hash] = faces
            return
        FACE_STORE['static_faces'][frame_hash] = faces


def clear_static_faces() -> None:
    FACE_STORE['static_faces'] = {}
    FACE_STORE_2['static_faces'] = {}


def create_frame_hash(vision_frame: VisionFrame) -> Optional[str]:
    return hashlib.sha1(vision_frame.tobytes()).hexdigest() if numpy.any(vision_frame) else None


def get_reference_faces(is_face_swapper: bool = False) -> Tuple[Optional[FaceSet], Optional[FaceSet]]:
    from facefusion.face_analyser import get_avg_faces

    set_out = {}
    set_out_2 = {}
    all_faces = []
    all_faces_2 = []

    if not is_face_swapper and 'face_swapper' in state_manager.get_item('processors'):
        source_face, source_face_2 = get_avg_faces()
        if source_face:
            all_faces.append(source_face)
        if source_face_2:
            all_faces_2.append(source_face_2)
    reference_face_dict = state_manager.get_item('reference_face_dict')
    if reference_face_dict:
        for frame_number, faces in reference_face_dict.items():
            for face in faces:
                all_faces.append(face)
        set_out['reference_faces'] = all_faces
    reference_face_dict_2 = state_manager.get_item('reference_face_dict_2')
    if reference_face_dict_2:
        for frame_number, faces in reference_face_dict_2.items():
            for face in faces:
                all_faces_2.append(face)
        set_out_2['reference_faces'] = all_faces_2
    return set_out, set_out_2


def append_reference_face(name: str, face: Face, dict_2=False) -> None:
    if dict_2:
        if name not in FACE_STORE_2['reference_faces']:
            FACE_STORE_2['reference_faces'][name] = []
        FACE_STORE_2['reference_faces'][name].append(face)
        return
    if name not in FACE_STORE['reference_faces']:
        FACE_STORE['reference_faces'][name] = []
    FACE_STORE['reference_faces'][name].append(face)


def clear_reference_faces() -> None:
    FACE_STORE['reference_faces'] = {}
    FACE_STORE_2['reference_faces'] = {}
