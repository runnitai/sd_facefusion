import os
import threading
from argparse import ArgumentParser
from typing import Dict, Tuple, List

import PIL
import cv2
import numpy as np
from PIL import ImageOps
from PIL.Image import Image

from facefusion import logger, wording, state_manager, inference_manager
from facefusion.face_ana import warp_and_crop_face, get_reference_facial_points
from facefusion.face_analyser import get_many_faces
from facefusion.filesystem import is_image, is_video, resolve_relative_path
from facefusion.jobs.job_store import register_step_keys
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.typing import StyleChangerInputs
from facefusion.typing import (
    ProcessMode, VisionFrame, QueuePayload, ModelSet,
    ApplyStateItem, Args, InferencePool
)
from facefusion.vision import read_image, write_image

THREAD_LOCK: threading.Lock = threading.Lock()
NAME = __name__.upper()


def padTo16x(image):
    h, w, c = np.shape(image)
    if h % 16 == 0 and w % 16 == 0:
        return image, h, w
    nh, nw = (h // 16 + 1) * 16, (w // 16 + 1) * 16
    img_new = np.ones((nh, nw, 3), np.uint8) * 255
    img_new[:h, :w, :] = image
    return img_new, h, w


def load_image(image_path: str) -> Image:
    with open(image_path, 'rb') as f:
        img = PIL.Image.open(f)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')
    return img


def convert_to_ndarray(input) -> np.ndarray:
    if isinstance(input, str):
        img = np.array(load_image(input))
    elif isinstance(input, Image):
        img = np.array(input.convert('RGB'))
    elif isinstance(input, np.ndarray):
        if len(input.shape) == 2:
            input = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        img = input[:, :, ::-1]
    elif isinstance(input, Dict):
        img = input.get('image', None)
        if img:
            img = np.array(load_image(img))
    else:
        raise TypeError(f'input should be either str, PIL.Image, np.array, but got {type(input)}')
    return img


def pad_to_multiple_of(img, multiple):
    h, w, _ = img.shape
    nh, nw = ((h + multiple - 1) // multiple) * multiple, ((w + multiple - 1) // multiple) * multiple
    padded_img = np.zeros((nh, nw, 3), dtype=np.uint8)
    padded_img[:h, :w, :] = img
    return padded_img, h, w


def forward_head(session, head_img: np.ndarray) -> np.ndarray:
    processed = head_img[:, :, ::-1].astype(np.float32)
    return session.run(None, {"input_image:0": processed})[0]


def clear_inference_pool() -> None:
    head_context = f"{NAME}.head"
    bg_context = f"{NAME}.bg"
    inference_manager.clear_inference_pool(head_context)
    inference_manager.clear_inference_pool(bg_context)


class StyleChanger(BaseProcessor):
    def process_image(self, target_path: str, output_path: str) -> None:
        pass

    MODEL_SET: ModelSet = {
        'anime': {
            'hashes': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_h.hash',
                    'path': resolve_relative_path('../.assets/models/style/anime_h.hash')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_bg.hash',
                    'path': resolve_relative_path('../.assets/models/style/anime_bg.hash')
                }
            },
            'sources': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_h.onnx',
                    'path': resolve_relative_path('../.assets/models/style/anime_h.onnx')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_bg.onnx',
                    'path': resolve_relative_path('../.assets/models/style/anime_bg.onnx')
                }
            }
        },
        '3d': {
            'hashes': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_h.hash',
                    'path': resolve_relative_path('../.assets/models/style/3d_h.hash')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_bg.hash',
                    'path': resolve_relative_path('../.assets/models/style/3d_bg.hash')
                }
            },
            'sources': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_h.onnx',
                    'path': resolve_relative_path('../.assets/models/style/3d_h.onnx')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_bg.onnx',
                    'path': resolve_relative_path('../.assets/models/style/3d_bg.onnx')
                }
            }
        },
        'handdrawn': {
            'hashes': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_h.hash',
                    'path': resolve_relative_path('../.assets/models/style/handdrawn_h.hash')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_bg.hash',
                    'path': resolve_relative_path('../.assets/models/style/handdrawn_bg.hash')
                }
            },
            'sources': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_h.onnx',
                    'path': resolve_relative_path('../.assets/models/style/handdrawn_h.onnx')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_bg.onnx',
                    'path': resolve_relative_path('../.assets/models/style/handdrawn_bg.onnx')
                }
            }
        },
        'sketch': {
            'hashes': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_h.hash',
                    'path': resolve_relative_path('../.assets/models/style/sketch_h.hash')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_bg.hash',
                    'path': resolve_relative_path('../.assets/models/style/sketch_bg.hash')
                }
            },
            'sources': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_h.onnx',
                    'path': resolve_relative_path('../.assets/models/style/sketch_h.onnx')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_bg.onnx',
                    'path': resolve_relative_path('../.assets/models/style/sketch_bg.onnx')
                }
            }
        },
        'artstyle': {
            'hashes': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_h.hash',
                    'path': resolve_relative_path('../.assets/models/style/artstyle_h.hash')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_bg.hash',
                    'path': resolve_relative_path('../.assets/models/style/artstyle_bg.hash')
                }
            },
            'sources': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_h.onnx',
                    'path': resolve_relative_path('../.assets/models/style/artstyle_h.onnx')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_bg.onnx',
                    'path': resolve_relative_path('../.assets/models/style/artstyle_bg.onnx')
                }
            }
        },
        'design': {
            'hashes': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_h.hash',
                    'path': resolve_relative_path('../.assets/models/style/design_h.hash')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_bg.hash',
                    'path': resolve_relative_path('../.assets/models/style/design_bg.hash')
                }
            },
            'sources': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_h.onnx',
                    'path': resolve_relative_path('../.assets/models/style/design_h.onnx')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_bg.onnx',
                    'path': resolve_relative_path('../.assets/models/style/design_bg.onnx')
                }
            }
        },
        'illustration': {
            'hashes': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_h.hash',
                    'path': resolve_relative_path('../.assets/models/style/illustration_h.hash')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_bg.hash',
                    'path': resolve_relative_path('../.assets/models/style/illustration_bg.hash')
                }
            },
            'sources': {
                'head': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_h.onnx',
                    'path': resolve_relative_path('../.assets/models/style/illustration_h.onnx')
                },
                'background': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_bg.onnx',
                    'path': resolve_relative_path('../.assets/models/style/illustration_bg.onnx')
                }
            }
        },
        'alpha': {
            'internal': True,  # Internal model, not to be listed
            'hashes': {
                'image': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/alpha.hash',
                    'path': resolve_relative_path('../.assets/models/style/alpha.hash')
                }
            },
            'sources': {
                'image': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/alpha.jpg',
                    'path': resolve_relative_path('../.assets/models/style/alpha.jpg')
                }
            }
        }
    }
    priority = 1
    is_face_processor: bool = True
    model_key: str = 'style_changer_model'
    REFERENCE_PTS = get_reference_facial_points(default_square=True)
    BOX_WIDTH = 288
    STYLE_MODEL_DIR = resolve_relative_path('../.assets/models/style')
    GLOBAL_MASK = cv2.imread(os.path.join(STYLE_MODEL_DIR, 'alpha.jpg'))
    GLOBAL_MASK = cv2.resize(GLOBAL_MASK, (BOX_WIDTH, BOX_WIDTH), interpolation=cv2.INTER_AREA)
    GLOBAL_MASK = cv2.cvtColor(GLOBAL_MASK, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def __init__(self):
        super().__init__()
        self.reference_pts = get_reference_facial_points(default_square=True)
        self.box_width = 288
        self.global_mask = self._initialize_global_mask()
        self.model_path = resolve_relative_path('../.assets/models/style')

    @staticmethod
    def _initialize_global_mask() -> np.ndarray:
        style_model_dir = resolve_relative_path('../.assets/models/style')
        mask_path = os.path.join(style_model_dir, 'alpha.jpg')
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (288, 288), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def register_args(self, program: ArgumentParser) -> None:
        program.add_argument(
            '--style-changer-model',
            help=wording.get('help.style_changer_model'),
            default='3d',
            choices=self.list_models()
        )
        program.add_argument(
            '--style-changer-target',
            help=wording.get('help.style_changer_target'),
            default='target',
            choices=['source', 'target']
        )
        program.add_argument(
            '--style-changer-skip-head',
            help=wording.get('help.style_changer_skip_head'),
            action='store_true'
        )
        register_step_keys(['style_changer_model', 'style_changer_target', 'style_changer_skip_head'])

    def apply_args(self, args: Args, apply_state_item: ApplyStateItem) -> None:
        apply_state_item('style_changer_model', args.get('style_changer_model'))
        apply_state_item('style_changer_target', args.get('style_changer_target'))

    def pre_process(self, mode: ProcessMode) -> bool:
        target_path = state_manager.get_item('target_path')
        output_path = state_manager.get_item('output_path')
        if mode in ['output', 'preview'] and not is_image(target_path) and not is_video(target_path):
            logger.error(wording.get('select_image_or_video_target') + wording.get('exclamation_mark'), NAME)
            return False
        if mode == 'output' and not output_path:
            logger.error(wording.get('select_file_or_directory_output') + wording.get('exclamation_mark'), NAME)
            return False
        return True

    def get_inference_pool(self) -> InferencePool:
        model_opts = self.get_model_options().get("sources")
        head_sources = {"head":model_opts.get("head")}
        bg_sources = {"bg": model_opts.get("background")}
        head_context = f"{NAME}.head"
        bg_context = f"{NAME}.bg"
        head_pool = inference_manager.get_inference_pool(head_context, head_sources)
        bg_pool = inference_manager.get_inference_pool(bg_context, bg_sources)
        return head_pool, bg_pool

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        if 'target' not in state_manager.get_item('style_changer_target'):
            return []
        output_frames = []
        for payload in queue_payloads:
            frame_path = payload['frame_path']
            frame_number = payload['frame_number']
            vision_frame = read_image(frame_path)
            processed_frame = self.process_frame({'target_vision_frame': vision_frame})
            write_image(frame_path, processed_frame)
            output_frames.append((frame_number, frame_path))
        return output_frames

    def process_src_image(self, input_path: str, output_path: str) -> str:
        #input_file, input_extension = os.path.splitext(input_path)
        style_target = state_manager.get_item('style_changer_target')
        skip_head = state_manager.get_item('style_changer_skip_head') if style_target == 'source' else False
        skip_bg = "source head" in style_target
        result = self.change_style(input_path, skip_head=skip_head, skip_bg=skip_bg)
        cv2.imwrite(output_path, result)  # Convert back to BGR if needed
        print(f"Image processing complete. Output saved to {output_path}.")
        return output_path

    def process_frame(self, inputs: StyleChangerInputs) -> VisionFrame:
        vision_frame = inputs['target_vision_frame']
        style_target = state_manager.get_item('style_changer_target')
        skip_head = state_manager.get_item('style_changer_skip_head') if style_target == 'target' else True
        return self.change_style(vision_frame, skip_head)

    def change_style(self, temp_vision_frame: VisionFrame, skip_head: bool = False,
                     skip_bg: bool = False) -> VisionFrame:
        # Similar structure to face_swapper: get inference sessions
        head_pool, bg_pool = self.get_inference_pool()
        sess_head = head_pool.get("head")
        sess_bg = bg_pool.get("bg")

        img = convert_to_ndarray(temp_vision_frame)
        ori_h, ori_w, _ = img.shape

        img_resized = self.resize_size(img, size=720)
        img_bgr = img_resized[:, :, ::-1]

        # Background inference
        res = self.forward_bg_process(sess_bg, img_bgr) if not skip_bg else img_bgr

        # Faces and heads
        if not skip_head:
            landmarks_2 = get_many_faces([img])
            if landmarks_2 is not None and len(landmarks_2) > 0:
                for landmark in landmarks_2:
                    f5p = landmark.landmark_set.get('5')
                    head_img, trans_inv = warp_and_crop_face(
                        img_resized,
                        f5p,
                        ratio=0.75,
                        reference_pts=self.REFERENCE_PTS,
                        crop_size=(self.BOX_WIDTH, self.BOX_WIDTH),
                        return_trans_inv=True
                    )
                    head_res = self.forward_head_process(sess_head, head_img)

                    head_trans_inv = cv2.warpAffine(
                        head_res,
                        trans_inv, (img_resized.shape[1], img_resized.shape[0]),
                        borderValue=(0, 0, 0)
                    )
                    mask = self.GLOBAL_MASK
                    mask_trans_inv = cv2.warpAffine(
                        mask,
                        trans_inv, (img_resized.shape[1], img_resized.shape[0]),
                        borderValue=(0, 0, 0)
                    )
                    mask_trans_inv = cv2.resize(mask_trans_inv, (head_trans_inv.shape[1], head_trans_inv.shape[0]))
                    mask_trans_inv = np.expand_dims(mask_trans_inv, 2)

                    res = cv2.resize(res, (head_trans_inv.shape[1], head_trans_inv.shape[0]))
                    res = mask_trans_inv * head_trans_inv + (1 - mask_trans_inv) * res

        res = cv2.resize(res, (ori_w, ori_h), interpolation=cv2.INTER_AREA)
        return res

    @staticmethod
    def resize_image(img: np.ndarray, size: int) -> np.ndarray:
        h, w, _ = img.shape
        scale = min(size / h, size / w)
        resized_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized_img

    @staticmethod
    def resize_size(image, size=720):
        h, w, c = np.shape(image)
        if min(h, w) > size:
            if h > w:
                h, w = int(size * h / w), size
            else:
                h, w = size, int(size * w / h)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def forward_bg_process(bg_session, img_bgr: np.ndarray) -> np.ndarray:
        pad_bg, pad_h, pad_w = padTo16x(img_bgr)
        input_data = pad_bg.astype(np.float32)
        res = bg_session.run(None, {"input_image:0": input_data})[0]
        res = res[:pad_h, :pad_w, :]
        return res

    @staticmethod
    def forward_head_process(head_session, head_img: np.ndarray) -> np.ndarray:
        head_input = head_img[:, :, ::-1].astype(np.float32)
        result = head_session.run(None, {"input_image:0": head_input})[0]
        return result

    def blend_faces(self, resized_img, trans_inv, bg_result):
        mask = self.global_mask
        mask_inv = cv2.warpAffine(mask, trans_inv, (resized_img.shape[1], resized_img.shape[0]))
        return cv2.addWeighted(bg_result, 0.5, mask_inv, 0.5, 0)
