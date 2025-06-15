from typing import List

import numpy

from facefusion import state_manager
from facefusion.typing import Face, FaceSelectorOrder, FaceSet, Gender, Race


def find_similar_faces(faces: List[Face], reference_faces: FaceSet, face_distance: float) -> List[Face]:
    """
    Find similar faces with optional caching integration for video processing
    """
    # Try to use cached face matching if available
    try:
        from facefusion.video_face_index import VIDEO_FACE_INDEX
        from facefusion.face_analyser import get_many_faces
        
        # Check if we're in video processing context with frame number
        current_frame_number = getattr(get_many_faces, '_current_frame_number', None)
        target_path = state_manager.get_item('target_path')
        
        if current_frame_number is not None and target_path:
            # Try to get cached face matches
            cached_matches = VIDEO_FACE_INDEX.get_cached_face_matches(
                target_path, current_frame_number, faces, reference_faces, face_distance
            )
            
            if cached_matches is not None:
                return cached_matches
    except (ImportError, Exception):
        # If caching fails, continue with normal processing
        pass
    
    # Normal face matching logic
    similar_faces: List[Face] = []
    if faces and reference_faces:
        for reference_face in reference_faces:
            if not similar_faces:
                for face in faces:
                    if compare_faces(face, reference_face, face_distance):
                        similar_faces.append(face)
    
    # Try to cache the result if we're in video context
    try:
        from facefusion.video_face_index import VIDEO_FACE_INDEX
        from facefusion.face_analyser import get_many_faces
        
        current_frame_number = getattr(get_many_faces, '_current_frame_number', None)
        target_path = state_manager.get_item('target_path')
        
        if current_frame_number is not None and target_path:
            VIDEO_FACE_INDEX.cache_face_matches(
                target_path, current_frame_number, faces, reference_faces, face_distance, similar_faces
            )
    except (ImportError, Exception):
        pass
    
    return similar_faces


def compare_faces(face: Face, reference_face: Face, face_distance: float) -> bool:
    current_face_distance = calc_face_distance(face, reference_face)
    return current_face_distance < face_distance


def calc_face_distance(face: Face, reference_face: Face) -> float:
    if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
        return 1 - numpy.dot(face.normed_embedding, reference_face.normed_embedding)
    return 1


def sort_and_filter_faces(faces: List[Face], sorts=None) -> List[Face]:
    if not sorts:
        sorts = current_sort_values()
    if faces:
        if sorts.get('face_selector_order'):
            faces = sort_by_order(faces, state_manager.get_item('face_selector_order'))
        if sorts.get('face_selector_gender'):
            faces = filter_by_gender(faces, state_manager.get_item('face_selector_gender'))
        if sorts.get('face_selector_race'):
            faces = filter_by_race(faces, state_manager.get_item('face_selector_race'))
        age_start = sorts.get('face_selector_age_start')
        age_end = sorts.get('face_selector_age_end')
        if age_start == 0:
            age_start = None
        if age_end == 100:
            age_end = None
        if age_start or age_end:
            faces = filter_by_age(faces, age_start, age_end)
    return faces


def current_sort_values():
    keys = ['face_selector_order', 'face_selector_gender', 'face_selector_race', 'face_selector_age_start',
            'face_selector_age_end']
    sorts = {}
    for key in keys:
        sorts[key] = state_manager.get_item(key)
    return sorts


def sort_by_order(faces: List[Face], order: FaceSelectorOrder) -> List[Face]:
    if order == 'left-right':
        return sorted(faces, key=lambda face: face.bounding_box[0])
    if order == 'right-left':
        return sorted(faces, key=lambda face: face.bounding_box[0], reverse=True)
    if order == 'top-bottom':
        return sorted(faces, key=lambda face: face.bounding_box[1])
    if order == 'bottom-top':
        return sorted(faces, key=lambda face: face.bounding_box[1], reverse=True)
    if order == 'small-large':
        return sorted(faces, key=lambda face: (face.bounding_box[2] - face.bounding_box[0]) * (
                face.bounding_box[3] - face.bounding_box[1]))
    if order == 'large-small':
        return sorted(faces, key=lambda face: (face.bounding_box[2] - face.bounding_box[0]) * (
                face.bounding_box[3] - face.bounding_box[1]), reverse=True)
    if order == 'best-worst':
        return sorted(faces, key=lambda face: face.score_set.get('detector'), reverse=True)
    if order == 'worst-best':
        return sorted(faces, key=lambda face: face.score_set.get('detector'))
    return faces


def filter_by_gender(faces: List[Face], gender: Gender) -> List[Face]:
    if gender == 'none' or not gender:
        return faces
    filter_faces = []
    for face in faces:
        if face.gender == gender:
            filter_faces.append(face)
    return filter_faces


def filter_by_age(faces: List[Face], face_selector_age_start: int, face_selector_age_end: int) -> List[Face]:
    filter_faces = []
    if not face_selector_age_start:
        face_selector_age_start = 0
    if not face_selector_age_end:
        face_selector_age_end = 100
    age = range(face_selector_age_start, face_selector_age_end)

    for face in faces:
        if set(face.age) & set(age):
            filter_faces.append(face)
    return filter_faces


def filter_by_race(faces: List[Face], race: Race) -> List[Face]:
    if race == 'none' or not race:
        return faces
    filter_faces = []
    for face in faces:
        if face.race == race:
            filter_faces.append(face)
    return filter_faces
