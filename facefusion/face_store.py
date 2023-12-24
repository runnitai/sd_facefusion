from typing import Optional, List
import hashlib

import facefusion.globals
from facefusion.typing import Frame, Face, FaceStore, FaceSet

FACE_STORE: FaceStore = \
    {
        'static_faces': {},
        'reference_faces': {}
    }


def get_static_faces(frame: Frame) -> Optional[List[Face]]:
    frame_hash = create_frame_hash(frame)
    if frame_hash in FACE_STORE['static_faces']:
        return FACE_STORE['static_faces'][frame_hash]
    return None


def set_static_faces(frame: Frame, faces: List[Face]) -> None:
    frame_hash = create_frame_hash(frame)
    if frame_hash:
        FACE_STORE['static_faces'][frame_hash] = faces


def clear_static_faces() -> None:
    FACE_STORE['static_faces'] = {}


def create_frame_hash(frame: Frame) -> Optional[str]:
    return hashlib.sha1(frame.tobytes()).hexdigest() if frame.any() else None


def get_reference_faces_original() -> Optional[FaceSet]:
    if FACE_STORE['reference_faces']:
        return FACE_STORE['reference_faces']
    return None


def get_reference_faces() -> Optional[FaceSet]:
    set_out = {}
    all_faces = []
    for frame_number, faces in facefusion.globals.reference_face_dict.items():
        for face in faces:
            all_faces.append(face)
    set_out['reference_faces'] = all_faces
    return set_out


def append_reference_face(name: str, face: Face) -> None:
    if name not in FACE_STORE['reference_faces']:
        FACE_STORE['reference_faces'][name] = []
    FACE_STORE['reference_faces'][name].append(face)


def clear_reference_faces() -> None:
    FACE_STORE['reference_faces'] = {}
