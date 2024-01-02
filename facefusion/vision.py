from typing import Optional, List
from functools import lru_cache
import cv2

from facefusion.typing import Frame

LAST_VIDEO_PATH: Optional[str] = None
LAST_FPS: Optional[float] = None
LAST_TOTAL_FRAMES: Optional[int] = None


def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Frame]:
    if video_path:
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            frame_total = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))
            has_frame, frame = video_capture.read()
            video_capture.release()
            if has_frame:
                return frame
    return None


def detect_fps(video_path: str) -> Optional[float]:
    global LAST_VIDEO_PATH, LAST_FPS
    if video_path:
        if LAST_VIDEO_PATH == video_path and LAST_FPS is not None:
            return LAST_FPS
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            LAST_FPS = video_capture.get(cv2.CAP_PROP_FPS)
            LAST_VIDEO_PATH = video_path
            video_capture.release()
            return LAST_FPS
    LAST_VIDEO_PATH = None
    LAST_FPS = None
    return None


def count_video_frame_total(video_path: str) -> int:
    global LAST_VIDEO_PATH, LAST_TOTAL_FRAMES
    if video_path:
        if LAST_VIDEO_PATH == video_path and LAST_TOTAL_FRAMES is not None:
            return LAST_TOTAL_FRAMES
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            video_frame_total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_capture.release()
            LAST_VIDEO_PATH = video_path
            LAST_TOTAL_FRAMES = video_frame_total
            return video_frame_total
    LAST_VIDEO_PATH = None
    LAST_TOTAL_FRAMES = None
    return 0


def normalize_frame_color(frame: Frame) -> Frame:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_frame_dimension(frame: Frame, max_width: int, max_height: int) -> Frame:
    height, width = frame.shape[:2]
    if height > max_height or width > max_width:
        scale = min(max_height / height, max_width / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))
    return frame


@lru_cache(maxsize=128)
def read_static_image(image_path: str) -> Optional[Frame]:
    return read_image(image_path)


def read_static_images(image_paths: List[str]) -> Optional[List[Frame]]:
    frames = []
    if image_paths:
        for image_path in image_paths:
            frames.append(read_static_image(image_path))
    return frames


def read_image(image_path: str) -> Optional[Frame]:
    if image_path:
        return cv2.imread(image_path)
    return None


def write_image(image_path: str, frame: Frame) -> bool:
    if image_path:
        return cv2.imwrite(image_path, frame)
    return False
