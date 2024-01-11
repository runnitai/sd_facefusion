from typing import List, Optional

from facefusion.processors.frame.typings import FaceSwapperModel, FaceEnhancerModel, FrameEnhancerModel, \
    FaceDebuggerItem

face_swapper_model: Optional[FaceSwapperModel] = "inswapper_128_fp16"
face_enhancer_model: Optional[FaceEnhancerModel] = "codeformer"
face_enhancer_blend: Optional[int] = 100
frame_enhancer_model: Optional[FrameEnhancerModel] = "real_esrgan_x4plus"
frame_enhancer_blend: Optional[int] = 100
face_debugger_items: Optional[List[FaceDebuggerItem]] = ['bbox', 'kps', 'face-mask', 'score', 'distance']
