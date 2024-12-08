import argparse
import os
import threading
from typing import Any, Dict, Tuple
from typing import Optional, List

import PIL
import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageOps
from PIL.Image import Image
import facefusion.jobs.job_store
import facefusion.globals
import facefusion.processors.core as frame_processors
from facefusion import config, logger, wording, state_manager
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.face_ana import get_f5p, warp_and_crop_face, get_reference_facial_points, FaceAna
from facefusion.filesystem import is_image, is_video, resolve_relative_path
from facefusion.processors.typing import StyleChangerInputs
from facefusion.typing import ProcessMode, OptionsWithModel, VisionFrame, QueuePayload, Face, ModelSet, ModelOptions, \
    ApplyStateItem, Args
from facefusion.vision import read_image, write_image, read_static_image

FRAME_PROCESSOR = None
SELECTED_MODEL = None

THREAD_LOCK: threading.Lock = threading.Lock()
NAME = __name__.upper()

# Available models
MODEL_SET: ModelSet = \
    {
        'anime_bg': {
            'hashes': {
                'model': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_bg.hash',
                    'path': resolve_relative_path('../.assets/models/style/anime_bg.hash')
                }
            },
            'sources': {
                'model': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_bg.pb',
                    'path': resolve_relative_path('../.assets/models/style/anime_bg.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_h.pb',
                    'path': resolve_relative_path('../.assets/models/style/anime_h.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_bg.pb',
                    'path': resolve_relative_path('../.assets/models/style/3d_bg.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_h.pb',
                    'path': resolve_relative_path('../.assets/models/style/3d_h.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_bg.pb',
                    'path': resolve_relative_path('../.assets/models/style/handdrawn_bg.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_h.pb',
                    'path': resolve_relative_path('../.assets/models/style/handdrawn_h.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_bg.pb',
                    'path': resolve_relative_path('../.assets/models/style/sketch_bg.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_h.pb',
                    'path': resolve_relative_path('../.assets/models/style/sketch_h.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_bg.pb',
                    'path': resolve_relative_path('../.assets/models/style/artstyle_bg.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_h.pb',
                    'path': resolve_relative_path('../.assets/models/style/artstyle_h.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_bg.pb',
                    'path': resolve_relative_path('../.assets/models/style/design_bg.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_h.pb',
                    'path': resolve_relative_path('../.assets/models/style/design_h.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_bg.pb',
                    'path': resolve_relative_path('../.assets/models/style/illustration_bg.pb')
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
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_h.pb',
                    'path': resolve_relative_path('../.assets/models/style/illustration_h.pb')
                }
            }
        },
        'genshen_bg': {
            'hashes': {
                'model': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/genshen_bg.hash',
                    'path': resolve_relative_path('../.assets/models/style/genshen_bg.hash')
                }
            },
            'sources': {
                'model': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/genshen_bg.pb',
                    'path': resolve_relative_path('../.assets/models/style/genshen_bg.pb')
                }
            }
        },
        'genshen_h': {
            'hashes': {
                'model': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/genshen_h.hash',
                    'path': resolve_relative_path('../.assets/models/style/genshen_h.hash')
                }
            },
            'sources': {
                'model': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/genshen_h.pb',
                    'path': resolve_relative_path('../.assets/models/style/genshen_h.pb')
                }
            }
        },
        'anime2_bg': {
            'hashes': {
                'model': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime2_bg.hash',
                    'path': resolve_relative_path('../.assets/models/style/anime2_bg.hash')
                }
            },
            'sources': {
                'model': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime2_bg.pb',
                    'path': resolve_relative_path('../.assets/models/style/anime2_bg.pb')
                }
            }
        },
        'anime2_h': {
            'hashes': {
                'model': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime2_h.hash',
                    'path': resolve_relative_path('../.assets/models/style/anime2_h.hash')
                }
            },
            'sources': {
                'model': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime2_h.pb',
                    'path': resolve_relative_path('../.assets/models/style/anime2_h.pb')
                }
            }
        }
    }


OPTIONS: Optional[OptionsWithModel] = None

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()


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
        raise TypeError(f'input should be either str, PIL.Image,'
                        f' np.array, but got {type(input)}')
    return img


class ImageCartoonPipelineCustom:

    def __init__(self, model: str):
        """
        Initialize the cartoon pipeline.
        Args:
            model: Model ID to load specific style models.
        """
        # Define model paths
        style_model_dir = resolve_relative_path('../.assets/models/style')
        model_head_path = os.path.join(style_model_dir, f"{model}_h.pb")
        model_bg_path = os.path.join(style_model_dir, f"{model}_bg.pb")

        # Load models for head and background processing
        self.facer = FaceAna(style_model_dir)
        with tf.Graph().as_default():
            self.sess_anime_head = self.load_sess(model_head_path, 'model_anime_head')
            self.sess_anime_bg = self.load_sess(model_bg_path, 'model_anime_bg')

        # Configuration for masks and dimensions
        self.box_width = 288
        global_mask = cv2.imread(os.path.join(style_model_dir, 'alpha.jpg'))
        global_mask = cv2.resize(global_mask, (self.box_width, self.box_width), interpolation=cv2.INTER_AREA)
        self.global_mask = cv2.cvtColor(global_mask, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    @staticmethod
    def load_sess(model_path: str, name: str) -> tf.Session:
        """
        Load a TensorFlow model into a session.
        Args:
            model_path: Path to the model file.
            name: Name for the imported graph.
        Returns:
            A TensorFlow session with the loaded model.
        """
        model_config = tf.ConfigProto(allow_soft_placement=True)
        model_config.gpu_options.allow_growth = True
        sess = tf.Session(config=model_config)
        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name=name)
            sess.run(tf.global_variables_initializer())
        return sess

    def detect_face(self, img: np.ndarray) -> Any:
        """
        Detect faces in the input image using the FaceAna utility.
        Args:
            img: Input image in np.ndarray format.
        Returns:
            Landmarks of detected faces or None if no face is detected.
        """
        boxes, landmarks, _ = self.facer.run(img)
        if boxes.shape[0] == 0:
            return None
        return landmarks

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Process an image through the cartoonization pipeline.
        Args:
            img: Input image in np.ndarray format (RGB).
        Returns:
            Processed cartoonized image in np.ndarray format (RGB).
        """
        # Original dimensions
        img = convert_to_ndarray(img)
        ori_h, ori_w, _ = img.shape

        # Preprocessing: Resize for processing
        img_resized = resize_size(img, size=720)
        img_bgr = img_resized[:, :, ::-1]  # Convert RGB to BGR for OpenCV

        # Background processing
        pad_bg, pad_h, pad_w = padTo16x(img_bgr)
        bg_res = self.sess_anime_bg.run(
            self.sess_anime_bg.graph.get_tensor_by_name('model_anime_bg/output_image:0'),
            feed_dict={'model_anime_bg/input_image:0': pad_bg}
        )
        res = bg_res[:pad_h, :pad_w, :]

        # Face detection and processing
        landmarks = self.detect_face(img_resized)
        if landmarks is not None:
            for landmark in landmarks:
                # Get facial 5 points for alignment
                f5p = get_f5p(landmark, img_bgr)

                # Face alignment
                head_img, trans_inv = warp_and_crop_face(
                    img_resized,
                    f5p,
                    ratio=0.75,
                    reference_pts=get_reference_facial_points(default_square=True),
                    crop_size=(self.box_width, self.box_width),
                    return_trans_inv=True
                )

                # Head processing
                head_res = self.sess_anime_head.run(
                    self.sess_anime_head.graph.get_tensor_by_name('model_anime_head/output_image:0'),
                    feed_dict={'model_anime_head/input_image:0': head_img[:, :, ::-1]}
                )

                # Merge head and background
                head_trans_inv = cv2.warpAffine(
                    head_res,
                    trans_inv, (img.shape[1], img.shape[0]),
                    borderValue=(0, 0, 0)
                )
                mask = self.global_mask
                mask_trans_inv = cv2.warpAffine(
                    mask,
                    trans_inv, (img.shape[1], img.shape[0]),
                    borderValue=(0, 0, 0)
                )
                mask_trans_inv = np.expand_dims(mask_trans_inv, 2)
                res = mask_trans_inv * head_trans_inv + (1 - mask_trans_inv) * res

        # Postprocessing: Resize back to original dimensions
        res = cv2.resize(res, (ori_w, ori_h), interpolation=cv2.INTER_AREA)
        # res = res[:, :, ::-1]  # Convert BGR to RGB
        return np.clip(res, 0, 255).astype(np.uint8)


def get_frame_processor() -> Any:
    global FRAME_PROCESSOR, SELECTED_MODEL

    with THREAD_LOCK:
        selected_model = state_manager.get_item('style_changer_model')
        if FRAME_PROCESSOR is None or selected_model != SELECTED_MODEL:
            print(f"Loading style changer model: {selected_model}")
            FRAME_PROCESSOR = ImageCartoonPipelineCustom(model=selected_model)
            SELECTED_MODEL = selected_model
    return FRAME_PROCESSOR


def clear_frame_processor() -> None:
    global FRAME_PROCESSOR
    if FRAME_PROCESSOR is not None:
        print("Deleting frame processor")
        del FRAME_PROCESSOR
        print("Frame processor deleted")
    FRAME_PROCESSOR = None


def get_model_options() -> Tuple[ModelOptions, ModelOptions]:
    style_changer_model = state_manager.get_item('style_changer_model')
    head_model = MODEL_SET.get(f'{style_changer_model}_h')
    bg_model = MODEL_SET.get(f'{style_changer_model}_bg')
    return head_model, bg_model


def model_names() -> List[str]:
    names = []
    for key in MODEL_SET.keys():
        model_name = key.split('_')[0]
        if model_name not in names:
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


# Apply command-line arguments
def apply_args(args: Args, apply_state_item: ApplyStateItem) -> None:
    apply_state_item('style_changer_model', args.get('style_changer_model'))
    apply_state_item('style_changer_target', args.get('style_changer_target'))


# Pre-check before processing
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
    all_hashes = {**head_model_hashes, **bg_model_hashes}
    all_sources = {**head_model_sources, **bg_model_sources}
    return conditional_download_hashes(download_directory_path, all_hashes) and conditional_download_sources(
        download_directory_path, all_sources)


# Post-check after setup (not used here, but included for completeness)
def post_check() -> bool:
    return True


def pre_process(mode: ProcessMode) -> bool:
    if mode in ['output', 'preview'] and not is_image(facefusion.globals.target_path) and not is_video(
            facefusion.globals.target_path):
        logger.error(wording.get('select_image_or_video_target') + wording.get('exclamation_mark'), NAME)
        return False
    if mode == 'output' and not facefusion.globals.output_path:
        logger.error(wording.get('select_file_or_directory_output') + wording.get('exclamation_mark'), NAME)
        return False
    return True


def change_style(temp_vision_frame: VisionFrame) -> VisionFrame:
    frame_processor = get_frame_processor()
    img = frame_processor(img=temp_vision_frame)

    # Fallback to input frame if no output is generated
    if img is None:
        print("No processed image returned; using input frame.")
        return temp_vision_frame

    return img


def get_reference_frame(source_face: Face, target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
    if state_manager.get_item('style_target') == 'source':
        return temp_vision_frame
    return change_style(temp_vision_frame)


# Process an image
def process_src_image(input_path: str, style: str):
    # Output path is the input path, with _style_{style} appended to the file name
    input_file, input_extension = os.path.splitext(input_path)
    output_path = f"{input_file}_style_{style}{input_extension}"
    processor = get_frame_processor()
    result = processor(input_path)
    cv2.imwrite(output_path, result)
    print(f"Image processing complete. Output saved to {output_path}.")
    return output_path


# Post-process results (placeholder for cleanup actions)
def post_process() -> None:
    pass


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
        output_vision_frame = process_frame(
            {
                'target_vision_frame': target_vision_frame
            })
        write_image(target_vision_path, output_vision_frame)
        output_frames.append((target_frame_number, target_vision_path))
    return output_frames


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
    if state_manager.get_item('style_changer_target') == 'source':
        return
    target_vision_frame = read_static_image(target_path)
    output_vision_frame = process_frame(
        {
            'target_vision_frame': target_vision_frame
        })
    write_image(output_path, output_vision_frame)


def process_video(source_paths: List[str], source_paths_2: List[str], temp_frame_paths: List[str]) -> None:
    print("Processing video frames with style changer.")
    frame_processors.multi_process_frames(temp_frame_paths, process_frames)


def get_inference_pool() -> List[str]:
    return


def clear_inference_pool() -> None:
    clear_frame_processor()
