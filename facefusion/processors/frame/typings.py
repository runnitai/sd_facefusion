from typing import Literal, TypedDict

from facefusion.typing import Face, FaceSet, AudioFrame, VisionFrame

FaceDebuggerItem = Literal[
    'bounding-box', 'face-landmark-5', 'face-landmark-5/68', 'face-landmark-68', 'face-mask', 'face-detector-score', 'face-landmarker-score', 'age', 'gender']
FaceEnhancerModel = Literal[
    'codeformer', 'gfpgan_1.2', 'gfpgan_1.3', 'gfpgan_1.4', 'gpen_bfr_256', 'gpen_bfr_512', 'restoreformer_plus_plus']
FaceSwapperModel = Literal[
    'blendswap_256', 'ghost_unet_1_block', 'ghost_unet_2_block', 'ghost_unet_3_block', 'inswapper_128', 'inswapper_128_fp16', 'simswap_256', 'simswap_512_unofficial', 'uniface_256']
FrameEnhancerModel = Literal['lsdir_x4', 'nomos8k_sc_x4', 'real_esrgan_x4', 'real_esrgan_x4_fp16', 'span_kendata_x4']
LipSyncerModel = Literal['wav2lip_gan']

FaceDebuggerInputs = TypedDict('FaceDebuggerInputs',
                               {
                                   'reference_faces': FaceSet,
                                   'reference_faces_2': FaceSet,
                                   'target_vision_frame': VisionFrame,
                                   'source_frame': VisionFrame,
                                   'target_frame_number': int
                               })
FaceEnhancerInputs = TypedDict('FaceEnhancerInputs',
                               {
                                   'reference_faces': FaceSet,
                                   'reference_faces_2': FaceSet,
                                   'target_vision_frame': VisionFrame
                               })
FaceSwapperInputs = TypedDict('FaceSwapperInputs',
                              {
                                  'reference_faces': FaceSet,
                                  'reference_faces_2': FaceSet,
                                  'source_face': Face,
                                  'source_face_2': Face,
                                  'target_vision_frame': VisionFrame,
                                  'target_frame_number': int
                              })
FrameEnhancerInputs = TypedDict('FrameEnhancerInputs',
                                {
                                    'target_vision_frame': VisionFrame
                                })

StyleChangerInputs = TypedDict('StyleChangerInputs',
                                {
                                    'target_vision_frame': VisionFrame
                                })
LipSyncerInputs = TypedDict('LipSyncerInputs',
                            {
                                'reference_faces': FaceSet,
                                'reference_faces_2': FaceSet,
                                'source_audio_frame': AudioFrame,
                                'target_vision_frame': VisionFrame
                            })
