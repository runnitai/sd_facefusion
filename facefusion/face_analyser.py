from typing import List, Optional

import numpy

from facefusion import state_manager, logger
from facefusion.common_helper import get_first
from facefusion.face_helper import apply_nms, convert_to_face_landmark_5, estimate_face_angle, get_nms_threshold
from facefusion.face_store import get_static_faces, set_static_faces
from facefusion.typing import BoundingBox, Face, FaceLandmark5, FaceLandmarkSet, FaceScoreSet, Score, VisionFrame
from facefusion.vision import read_static_images
from facefusion.workers.classes.face_classifier import FaceClassifier
from facefusion.workers.classes.face_detector import FaceDetector
from facefusion.workers.classes.face_landmarker import FaceLandmarker
from facefusion.workers.classes.face_recognizer import FaceRecognizer

AVG_FACE_1: Optional[Face] = None
AVG_FACE_2: Optional[Face] = None
SOURCE_FRAMES_1: Optional[List[str]] = None
SOURCE_FRAMES_2: Optional[List[str]] = None
SOURCE_FRAME_DICT: Optional[dict] = {}
AVERAGE_FACE_DICT: Optional[dict] = {}
landmarker = FaceLandmarker()
detector = FaceDetector()
recognizer = FaceRecognizer()
classifier = FaceClassifier()


def create_faces(vision_frame: VisionFrame, bounding_boxes: List[BoundingBox], face_scores: List[Score],
                 face_landmarks_5: List[FaceLandmark5]) -> List[Face]:
    faces = []
    nms_threshold = get_nms_threshold(state_manager.get_item('face_detector_model'),
                                      state_manager.get_item('face_detector_angles'))
    keep_indices = apply_nms(bounding_boxes, face_scores, state_manager.get_item('face_detector_score'), nms_threshold)

    for index in keep_indices:
        bounding_box = bounding_boxes[index]
        face_score = face_scores[index]
        face_landmark_5 = face_landmarks_5[index]
        face_landmark_5_68 = face_landmark_5
        face_landmark_68_5 = landmarker.estimate_face_landmark_68_5(face_landmark_5_68)
        face_landmark_68 = face_landmark_68_5
        face_landmark_score_68 = 0.0
        face_angle = estimate_face_angle(face_landmark_68_5)

        if state_manager.get_item('face_landmarker_score') > 0:
            face_landmark_68, face_landmark_score_68 = landmarker.detect_face_landmarks(vision_frame, bounding_box,
                                                                                        face_angle)
        if face_landmark_score_68 > state_manager.get_item('face_landmarker_score'):
            face_landmark_5_68 = convert_to_face_landmark_5(face_landmark_68)

        face_landmark_set: FaceLandmarkSet = \
            {
                '5': face_landmark_5,
                '5/68': face_landmark_5_68,
                '68': face_landmark_68,
                '68/5': face_landmark_68_5
            }
        face_score_set: FaceScoreSet = \
            {
                'detector': face_score,
                'landmarker': face_landmark_score_68
            }
        embedding, normed_embedding = recognizer.calc_embedding(vision_frame, face_landmark_set.get('5/68'))
        gender, age, race = classifier.classify_face(vision_frame, face_landmark_set.get('5/68'))
        faces.append(Face(
            bounding_box=bounding_box,
            score_set=face_score_set,
            landmark_set=face_landmark_set,
            angle=face_angle,
            embedding=embedding,
            normed_embedding=normed_embedding,
            gender=gender,
            age=age,
            race=race
        ))
    return faces


def get_one_face(faces: List[Face], position: int = 0) -> Optional[Face]:
    if faces:
        position = min(position, len(faces) - 1)
        return faces[position]
    return None


def get_average_face(faces: List[Face]) -> Optional[Face]:
    embeddings = []
    normed_embeddings = []

    if faces:
        first_face = get_first(faces)

        for face in faces:
            embeddings.append(face.embedding)
            normed_embeddings.append(face.normed_embedding)

        return Face(
            bounding_box=first_face.bounding_box,
            score_set=first_face.score_set,
            landmark_set=first_face.landmark_set,
            angle=first_face.angle,
            embedding=numpy.mean(embeddings, axis=0),
            normed_embedding=numpy.mean(normed_embeddings, axis=0),
            gender=first_face.gender,
            age=first_face.age,
            race=first_face.race
        )
    return None


# Define a global cache dictionary
vision_frame_cache = {}


def get_frame_hash(vision_frame: VisionFrame) -> int:
    """
    Compute a unique hash for the VisionFrame using its contents.
    """
    # Use hash of the flattened array's bytes for performance
    return hash(vision_frame.tobytes())


def get_many_faces(vision_frames: List[VisionFrame]) -> List[Face]:
    many_faces: List[Face] = []

    for vision_frame in vision_frames:
        if numpy.any(vision_frame):  # Ensure the frame is not empty
            # Compute hash for the current vision frame
            frame_hash = get_frame_hash(vision_frame)
            # Check if faces for this frame are already cached
            if frame_hash in vision_frame_cache:
                many_faces.extend(vision_frame_cache[frame_hash])
                continue

            # Process the frame to detect faces
            static_faces = get_static_faces(vision_frame)
            if static_faces:
                many_faces.extend(static_faces)
                vision_frame_cache[frame_hash] = static_faces  # Cache the result
            else:
                all_bounding_boxes = []
                all_face_scores = []
                all_face_landmarks_5 = []

                for face_detector_angle in state_manager.get_item('face_detector_angles'):
                    if face_detector_angle == 0:
                        bounding_boxes, face_scores, face_landmarks_5 = detector.detect_faces(vision_frame)
                    else:
                        bounding_boxes, face_scores, face_landmarks_5 = detector.detect_rotated_faces(vision_frame,
                                                                                                      face_detector_angle)
                    all_bounding_boxes.extend(bounding_boxes)
                    all_face_scores.extend(face_scores)
                    all_face_landmarks_5.extend(face_landmarks_5)

                if all_bounding_boxes and all_face_scores and all_face_landmarks_5 and state_manager.get_item(
                        'face_detector_score') > 0:
                    faces = create_faces(vision_frame, all_bounding_boxes, all_face_scores, all_face_landmarks_5)
                    if faces:
                        many_faces.extend(faces)
                        set_static_faces(vision_frame, faces)
                        vision_frame_cache[frame_hash] = faces  # Cache the result

    return many_faces


def get_average_faces():
    global SOURCE_FRAME_DICT, AVERAGE_FACE_DICT

    # Ensure SOURCE_FRAME_DICT and AVERAGE_FACE_DICT are initialized
    if SOURCE_FRAME_DICT is None:
        SOURCE_FRAME_DICT = {}
    if AVERAGE_FACE_DICT is None:
        AVERAGE_FACE_DICT = {}

    source_frame_dict = state_manager.get_item('source_frame_dict') or {}
    new_average_face_dict = {}
    # Make a copy of source_frame_dict so we avoid issues with the original changing during iteration
    source_frame_dict = source_frame_dict.copy()
    for source_face_index, frame_paths in source_frame_dict.items():
        # Retrieve the cached frame paths for the current index
        cached_paths = SOURCE_FRAME_DICT.get(source_face_index, [])

        # Only process if the frame paths have changed or are new
        if cached_paths != frame_paths or source_face_index not in AVERAGE_FACE_DICT:
            faces = []

            for frame_path in frame_paths:
                try:
                    frames = read_static_images([frame_path])
                    if not frames:
                        logger.warn(f"No frames found for path: {frame_path}", __name__)
                        continue

                    face = get_one_face(get_many_faces(frames))
                    if face:
                        faces.append(face)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_path}: {e}", __name__)

            # Compute the average face only if faces were detected
            if faces:
                new_average_face_dict[source_face_index] = get_average_face(faces)

        # Retain existing average faces if frame paths are unchanged
        elif source_face_index in AVERAGE_FACE_DICT:
            new_average_face_dict[source_face_index] = AVERAGE_FACE_DICT[source_face_index]

    # Update global dictionaries
    AVERAGE_FACE_DICT = new_average_face_dict
    SOURCE_FRAME_DICT = source_frame_dict

    return AVERAGE_FACE_DICT

