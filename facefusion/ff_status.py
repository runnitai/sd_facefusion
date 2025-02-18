import os.path
import time

from facefusion import state_manager
from facefusion.vision import count_video_frame_total, detect_video_fps


class FFStatus:
    _instance = None
    _is_initialized = False  # Ensure this is declared at the class level

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FFStatus, cls).__new__(cls)
        return cls._instance

    def __init__(self, re_init=False):
        # Check if this is the first time __init__ is called or if re-initialization is requested
        if not self._is_initialized or re_init:
            self.queue_total = 0
            self.queue_current = 0
            self.job_total = 0
            self.job_current = 0
            self.status = ""
            self.started = False
            self.cancelled = False
            self.time_start = None
            self.preview_image = None
            # Mark as initialized to prevent reinitialization, unless explicitly requested
            FFStatus._is_initialized = True

    def start(self, status: str = None):
        print(f"Starting FFStatus with 1 job.")
        """Start the status tracker with the first job in the queue"""
        self.queue_total = 1
        self.queue_current = 0
        self.preview_image = None
        self.status = status
        self.started = True
        self.time_start = time.time()
        if self.queue_total > 0:
            self.started = True
            self.cancelled = False
            self.job_total = self._compute_total_steps()
            self.job_current = 0
            print(f"Starting job with {self.job_total} steps")

    def update(self, status: str):
        """Update the current job status"""
        if self.preview_image and not os.path.exists(self.preview_image):
            self.preview_image = None
        self.status = status

    def update_preview(self, image: str):
        """Update the current job preview image"""
        if image and not os.path.exists(image):
            image = None
        self.preview_image = image

    def step(self, num_steps=1):
        """Increment the current job progress by num_steps"""
        self.job_current += num_steps
        if self.preview_image and not os.path.exists(self.preview_image):
            self.preview_image = None
        if self.job_current > self.job_total:
            self.job_total = self.job_current

    def next(self, status: str = None, step_queue=True):
        """Move to the next job in the queue"""
        if step_queue:
            self.queue_current += 1
        self.job_current = 0
        self.job_total = self._compute_total_steps()
        if self.preview_image and not os.path.exists(self.preview_image):
            self.preview_image = None
        self.status = status

    def cancel(self):
        """Cancel the current job"""
        self.cancelled = True
        self.started = False
        if self.preview_image and not os.path.exists(self.preview_image):
            self.preview_image = None
        self.status = f"Cancelled after {self.queue_current} jobs"
        self.queue_total = 0
        self.queue_current = 0
        self.job_total = 0
        self.job_current = 0
        self.time_start = None
        self.preview_image = None

    def finish(self, status: str = None):
        """Finish the current job"""
        self.started = False
        self.cancelled = False
        self.queue_total = 0
        self.queue_current = 0
        self.job_total = 0
        self.job_current = 0
        self.time_start = None
        self.preview_image = None
        self.status = ""
        if status:
            self.status = status

    @staticmethod
    def _compute_total_steps():
        target_path = state_manager.get_item('target_path')
        frame_processors = state_manager.get_item('processors')
        trim_frame_start = state_manager.get_item('trim_frame_start')
        trim_frame_end = state_manager.get_item('trim_frame_end')
        from facefusion.filesystem import is_video, is_image
        if is_video(target_path):
            original_fps = detect_video_fps(target_path)
            fps = state_manager.get_item('output_video_fps')
            video_frame_total = count_video_frame_total(target_path)
            if trim_frame_start is not None:
                video_frame_total -= trim_frame_start
            if trim_frame_end is not None:
                video_frame_total = trim_frame_end - (trim_frame_start if trim_frame_start is not None else 0)
            total_frames = video_frame_total / original_fps * fps
            print(f"Total frames: {total_frames}, execution providers: {len(frame_processors)}")
            total_steps = total_frames * len(frame_processors)
            total_steps += 2
            if not state_manager.get_item('skip_audio'):
                total_steps += 1
        elif is_image(target_path):
            total_steps = len(frame_processors) + 1
        else:
            total_steps = 0
        return int(total_steps)


def update_status(message: str, scope: str = 'FACEFUSION.CORE') -> None:
    print('[' + scope + '] ' + message)
