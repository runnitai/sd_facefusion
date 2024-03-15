from typing import List, Optional

from facefusion.processors.frame.typings import FaceSwapperModel, FaceEnhancerModel, FrameEnhancerModel, \
    FaceDebuggerItem, LipSyncerModel

face_debugger_items: Optional[List[FaceDebuggerItem]] = ['bounding-box', 'landmark-5', 'landmark-68', 'face-mask', 'score', 'age', 'gender']
face_enhancer_model: Optional[FaceEnhancerModel] = "gfpgan_1.4"
face_enhancer_blend: Optional[int] = 100
face_swapper_model: Optional[FaceSwapperModel] = "inswapper_128_fp16"
frame_enhancer_model: Optional[FrameEnhancerModel] = "real_esrgan_4x"
frame_enhancer_blend: Optional[int] = 100
lip_syncer_model: Optional[LipSyncerModel] = "wav2lip_gan"
