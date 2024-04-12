from typing import Optional, List, Tuple
import hashlib
import numpy

import facefusion.globals
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


def get_reference_faces_original() -> Optional[FaceSet]:
    if FACE_STORE['reference_faces']:
        return FACE_STORE['reference_faces']
    return None


def get_reference_faces() -> Tuple[Optional[FaceSet], Optional[FaceSet]]:
    from extensions.sd_facefusion.facefusion.face_analyser import get_average_face

    set_out = {}
    set_out_2 = {}
    all_faces = []
    for frame_number, faces in facefusion.globals.reference_face_dict.items():
        for face in faces:
            all_faces.append(face)
    set_out['reference_faces'] = all_faces
    all_faces = []
    for frame_number, faces in facefusion.globals.reference_face_dict_2.items():
        for face in faces:
            all_faces.append(face)
    set_out_2['reference_faces'] = all_faces
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
