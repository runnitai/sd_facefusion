import argparse
import os
import threading
from typing import Dict, Tuple
from typing import Optional, List

import PIL
import cv2
import numpy as np
from PIL import ImageOps
from PIL.Image import Image

import facefusion.globals
import facefusion.jobs.job_store
import facefusion.processors.core as frame_processors
from facefusion import logger, wording, state_manager, inference_manager
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.face_ana import warp_and_crop_face, get_reference_facial_points
from facefusion.face_analyser import get_many_faces
from facefusion.filesystem import is_image, is_video, resolve_relative_path
from facefusion.processors.typing import StyleChangerInputs
from facefusion.typing import ProcessMode, OptionsWithModel, VisionFrame, QueuePayload, Face, ModelSet, ModelOptions, \
    ApplyStateItem, Args, InferencePool
from facefusion.vision import read_image, write_image, read_static_image

THREAD_LOCK: threading.Lock = threading.Lock()
NAME = __name__.upper()

MODEL_SET: ModelSet = {
    'anime_bg': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_bg.hash',
                'path': resolve_relative_path('../.assets/models/style/anime_bg.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_bg.onnx',
                'path': resolve_relative_path('../.assets/models/style/anime_bg.onnx')
            }
        }
    },
    'anime_h': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_h.hash',
                'path': resolve_relative_path('../.assets/models/style/anime_h.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_h.onnx',
                'path': resolve_relative_path('../.assets/models/style/anime_h.onnx')
            }
        }
    },
    '3d_bg': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_bg.hash',
                'path': resolve_relative_path('../.assets/models/style/3d_bg.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_bg.onnx',
                'path': resolve_relative_path('../.assets/models/style/3d_bg.onnx')
            }
        }
    },
    '3d_h': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_h.hash',
                'path': resolve_relative_path('../.assets/models/style/3d_h.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_h.onnx',
                'path': resolve_relative_path('../.assets/models/style/3d_h.onnx')
            }
        }
    },
    'handdrawn_bg': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_bg.hash',
                'path': resolve_relative_path('../.assets/models/style/handdrawn_bg.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_bg.onnx',
                'path': resolve_relative_path('../.assets/models/style/handdrawn_bg.onnx')
            }
        }
    },
    'handdrawn_h': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_h.hash',
                'path': resolve_relative_path('../.assets/models/style/handdrawn_h.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_h.onnx',
                'path': resolve_relative_path('../.assets/models/style/handdrawn_h.onnx')
            }
        }
    },
    'sketch_bg': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_bg.hash',
                'path': resolve_relative_path('../.assets/models/style/sketch_bg.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_bg.onnx',
                'path': resolve_relative_path('../.assets/models/style/sketch_bg.onnx')
            }
        }
    },
    'sketch_h': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_h.hash',
                'path': resolve_relative_path('../.assets/models/style/sketch_h.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_h.onnx',
                'path': resolve_relative_path('../.assets/models/style/sketch_h.onnx')
            }
        }
    },
    'artstyle_bg': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_bg.hash',
                'path': resolve_relative_path('../.assets/models/style/artstyle_bg.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_bg.onnx',
                'path': resolve_relative_path('../.assets/models/style/artstyle_bg.onnx')
            }
        }
    },
    'artstyle_h': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_h.hash',
                'path': resolve_relative_path('../.assets/models/style/artstyle_h.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_h.onnx',
                'path': resolve_relative_path('../.assets/models/style/artstyle_h.onnx')
            }
        }
    },
    'design_bg': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_bg.hash',
                'path': resolve_relative_path('../.assets/models/style/design_bg.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_bg.onnx',
                'path': resolve_relative_path('../.assets/models/style/design_bg.onnx')
            }
        }
    },
    'design_h': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_h.hash',
                'path': resolve_relative_path('../.assets/models/style/design_h.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_h.onnx',
                'path': resolve_relative_path('../.assets/models/style/design_h.onnx')
            }
        }
    },
    'illustration_bg': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_bg.hash',
                'path': resolve_relative_path('../.assets/models/style/illustration_bg.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_bg.onnx',
                'path': resolve_relative_path('../.assets/models/style/illustration_bg.onnx')
            }
        }
    },
    'illustration_h': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_h.hash',
                'path': resolve_relative_path('../.assets/models/style/illustration_h.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_h.onnx',
                'path': resolve_relative_path('../.assets/models/style/illustration_h.onnx')
            }
        }
    },
    'alpha': {
        'hashes': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/alpha.hash',
                'path': resolve_relative_path('../.assets/models/style/alpha.hash')
            }
        },
        'sources': {
            'model': {
                'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/alpha.jpg',
                'path': resolve_relative_path('../.assets/models/style/alpha.jpg')
            }
        }
    }
}

OPTIONS: Optional[OptionsWithModel] = None


# Global initialization similar to face_swapper structure
REFERENCE_PTS = get_reference_facial_points(default_square=True)
BOX_WIDTH = 288
STYLE_MODEL_DIR = resolve_relative_path('../.assets/models/style')
GLOBAL_MASK = cv2.imread(os.path.join(STYLE_MODEL_DIR, 'alpha.jpg'))
GLOBAL_MASK = cv2.resize(GLOBAL_MASK, (BOX_WIDTH, BOX_WIDTH), interpolation=cv2.INTER_AREA)
GLOBAL_MASK = cv2.cvtColor(GLOBAL_MASK, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def padTo16x(image):
    h, w, c = np.shape(image)
    if h % 16 == 0 and w % 16 == 0:
        return image, h, w
    nh, nw = (h // 16 + 1) * 16, (w // 16 + 1) * 16
    img_new = np.ones((nh, nw, 3), np.uint8) * 255
    img_new[:h, :w, :] = image
    return img_new, h, w


def resize_size(image, size=720):
    h, w, c = np.shape(image)
    if min(h, w) > size:
        if h > w:
            h, w = int(size * h / w), size
        else:
            h, w = size, int(size * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return image


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


def get_model_options() -> Tuple[ModelOptions, ModelOptions]:
    style_changer_model = state_manager.get_item('style_changer_model')
    head_model = MODEL_SET.get(f'{style_changer_model}_h')
    bg_model = MODEL_SET.get(f'{style_changer_model}_bg')
    return head_model, bg_model


def model_names() -> List[str]:
    names = []
    for key in MODEL_SET.keys():
        model_name = key.split('_')[0]
        if model_name not in names and model_name != 'alpha':
            names.append(model_name)
    return names


def register_args(program: argparse.ArgumentParser) -> None:
    program.add_argument(
        '--style-changer-model',
        help=wording.get('help.style_changer_model'),
        default='3d',
        choices=model_names()
    )
    program.add_argument(
        '--style-changer-target',
        help=wording.get('help.style_changer_target'),
        default='target',
        choices=['source', 'target']
    )
    facefusion.jobs.job_store.register_step_keys(['style_changer_model', 'style_changer_target'])


def apply_args(args: Args, apply_state_item: ApplyStateItem) -> None:
    apply_state_item('style_changer_model', args.get('style_changer_model'))
    apply_state_item('style_changer_target', args.get('style_changer_target'))


def pre_check() -> bool:
    model_name = facefusion.globals.style_changer_model
    if model_name not in model_names():
        logger.error(f"Model '{model_name}' not found in available models.", "STYLE_CHANGER")
        return False
    download_directory_path = resolve_relative_path('../.assets/models/style')
    head_model_options, bg_model_options = get_model_options()
    head_model_hashes = head_model_options.get('hashes')
    head_model_sources = head_model_options.get('sources')
    bg_model_hashes = bg_model_options.get('hashes')
    bg_model_sources = bg_model_options.get('sources')
    alpha_hashes = MODEL_SET.get('alpha').get('hashes')
    alpha_sources = MODEL_SET.get('alpha').get('sources')
    all_hashes = {**head_model_hashes, **bg_model_hashes, **alpha_hashes}
    all_sources = {**head_model_sources, **bg_model_sources, **alpha_sources}
    return conditional_download_hashes(download_directory_path, all_hashes) and conditional_download_sources(
        download_directory_path, all_sources)


def post_check() -> bool:
    return True


def pre_process(mode: ProcessMode) -> bool:
    target_path = state_manager.get_item('target_path')
    output_path = state_manager.get_item('output_path')
    if mode in ['output', 'preview'] and not is_image(target_path) and not is_video(target_path):
        logger.error(wording.get('select_image_or_video_target') + wording.get('exclamation_mark'), NAME)
        return False
    if mode == 'output' and not output_path:
        logger.error(wording.get('select_file_or_directory_output') + wording.get('exclamation_mark'), NAME)
        return False
    return True


def get_inference_pool() -> InferencePool:
    head_model_opts, bg_model_opts = get_model_options()
    head_model_sources = head_model_opts.get('sources')
    bg_model_sources = bg_model_opts.get('sources')
    head_model_context = __name__ + '.' + state_manager.get_item('face_swapper_model') + '_head'
    bg_model_context = __name__ + '.' + state_manager.get_item('face_swapper_model') + '_bg'
    head_pool = inference_manager.get_inference_pool(head_model_context, head_model_sources)
    bg_pool = inference_manager.get_inference_pool(bg_model_context, bg_model_sources)
    return head_pool, bg_pool


def clear_inference_pool() -> None:
    head_model_context = __name__ + '.' + state_manager.get_item('face_swapper_model') + '_head'
    bg_model_context = __name__ + '.' + state_manager.get_item('face_swapper_model') + '_bg'
    inference_manager.clear_inference_pool(head_model_context)
    inference_manager.clear_inference_pool(bg_model_context)


def forward_bg_process(bg_session, img_bgr: np.ndarray) -> np.ndarray:
    pad_bg, pad_h, pad_w = padTo16x(img_bgr)
    input_data = pad_bg.astype(np.float32)
    res = bg_session.run(None, {"input_image:0": input_data})[0]
    res = res[:pad_h, :pad_w, :]
    return res


def forward_head_process(head_session, head_img: np.ndarray) -> np.ndarray:
    head_input = head_img[:, :, ::-1].astype(np.float32)
    result = head_session.run(None, {"input_image:0": head_input})[0]
    return result


def change_style(temp_vision_frame: VisionFrame) -> VisionFrame:
    # Similar structure to face_swapper: get inference sessions
    skip_head = state_manager.get_item('style_changer_skip_head')
    head_pool, bg_pool = get_inference_pool()
    sess_head = head_pool.get("model")
    sess_bg = bg_pool.get("model")

    img = convert_to_ndarray(temp_vision_frame)
    ori_h, ori_w, _ = img.shape

    img_resized = resize_size(img, size=720)
    img_bgr = img_resized[:, :, ::-1]

    # Background inference
    res = forward_bg_process(sess_bg, img_bgr)

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
                    reference_pts=REFERENCE_PTS,
                    crop_size=(BOX_WIDTH, BOX_WIDTH),
                    return_trans_inv=True
                )
                head_res = forward_head_process(sess_head, head_img)

                head_trans_inv = cv2.warpAffine(
                    head_res,
                    trans_inv, (img_resized.shape[1], img_resized.shape[0]),
                    borderValue=(0, 0, 0)
                )
                mask = GLOBAL_MASK
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


def get_reference_frame(source_face: Face, target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
    if state_manager.get_item('style_target') == 'source':
        return temp_vision_frame
    return change_style(temp_vision_frame)


def process_src_image(input_path: str, style: str):
    input_file, input_extension = os.path.splitext(input_path)
    output_path = f"{input_file}_style_{style}{input_extension}"
    result = change_style(input_path)
    cv2.imwrite(output_path, result[:, :, ::-1])  # Convert back to BGR if needed
    print(f"Image processing complete. Output saved to {output_path}.")
    return output_path


def post_process() -> None:
    clear_inference_pool()


def process_frame(inputs: StyleChangerInputs) -> VisionFrame:
    target_vision_frame = inputs.get('target_vision_frame')
    converted = change_style(target_vision_frame)
    return converted


def process_frames(queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
    if state_manager.get_item('style_changer_target') == 'source':
        print(f"Skipping processing for source target.")
        return
    output_frames = []
    for queue_payload in queue_payloads:
        target_vision_path = queue_payload['frame_path']
        target_frame_number = queue_payload['frame_number']
        target_vision_frame = read_image(target_vision_path)
        output_vision_frame = process_frame({'target_vision_frame': target_vision_frame})
        write_image(target_vision_path, output_vision_frame)
        output_frames.append((target_frame_number, target_vision_path))
    return output_frames


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
    if state_manager.get_item('style_changer_target') == 'source':
        return
    target_vision_frame = read_static_image(target_path)
    output_vision_frame = process_frame({'target_vision_frame': target_vision_frame})
    write_image(output_path, output_vision_frame)


def process_video(source_paths: List[str], source_paths_2: List[str], temp_frame_paths: List[str]) -> None:
    print("Processing video frames with style changer.")
    frame_processors.multi_process_frames(temp_frame_paths, process_frames)
