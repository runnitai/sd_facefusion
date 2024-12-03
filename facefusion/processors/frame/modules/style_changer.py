import argparse
import os
import threading
from typing import Any, Dict
from typing import Literal, Optional, List

import PIL
import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageOps
from PIL.Image import Image

import facefusion.globals
import facefusion.processors.frame.core as frame_processors
from facefusion import config, logger, wording
from facefusion.download import conditional_download
from facefusion.face_ana import get_f5p, warp_and_crop_face, get_reference_facial_points, FaceAna
from facefusion.filesystem import is_image, is_video, resolve_relative_path
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.processors.frame.typings import StyleChangerInputs
from facefusion.typing import ProcessMode, OptionsWithModel, VisionFrame, QueuePayload, UpdateProcess, Face
from facefusion.vision import read_image, write_image, read_static_image

FRAME_PROCESSOR = None
THREAD_LOCK: threading.Lock = threading.Lock()
NAME = __name__.upper()

# Available models
MODEL_MAPPING = {
    "anime": {
        "bg": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_bg.pb",
            "path": resolve_relative_path('../.assets/models/style/anime_bg.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_bg.hash"
        },
        "head": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_h.pb",
            "path": resolve_relative_path('../.assets/models/style/anime_h.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime_h.hash"
        }
    },
    "3d": {
        "bg": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_bg.pb",
            "path": resolve_relative_path('../.assets/models/style/3d_bg.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_bg.hash"
        },
        "head": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_h.pb",
            "path": resolve_relative_path('../.assets/models/style/3d_h.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/3d_h.hash"
        }
    },
    "handdrawn": {
        "bg": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_bg.pb",
            "path": resolve_relative_path('../.assets/models/style/handdrawn_bg.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_bg.hash"
        },
        "head": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_h.pb",
            "path": resolve_relative_path('../.assets/models/style/handdrawn_h.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/handdrawn_h.hash"
        }
    },
    "sketch": {
        "bg": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_bg.pb",
            "path": resolve_relative_path('../.assets/models/style/sketch_bg.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_bg.hash"
        },
        "head": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_h.pb",
            "path": resolve_relative_path('../.assets/models/style/sketch_h.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/sketch_h.hash"
        }
    },
    "artstyle": {
        "bg": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_bg.pb",
            "path": resolve_relative_path('../.assets/models/style/artstyle_bg.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_bg.hash"
        },
        "head": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_h.pb",
            "path": resolve_relative_path('../.assets/models/style/artstyle_h.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/artstyle_h.hash"
        }
    },
    "design": {
        "bg": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_bg.pb",
            "path": resolve_relative_path('../.assets/models/style/design_bg.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_bg.hash"
        },
        "head": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_h.pb",
            "path": resolve_relative_path('../.assets/models/style/design_h.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/design_h.hash"
        }
    },
    "illustration": {
        "bg": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_bg.pb",
            "path": resolve_relative_path('../.assets/models/style/illustration_bg.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_bg.hash"
        },
        "head": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_h.pb",
            "path": resolve_relative_path('../.assets/models/style/illustration_h.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/illustration_h.hash"
        }
    },
    "genshen": {
        "bg": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/genshen_bg.pb",
            "path": resolve_relative_path('../.assets/models/style/genshen_bg.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/genshen_bg.hash"
        },
        "head": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/genshen_h.pb",
            "path": resolve_relative_path('../.assets/models/style/genshen_h.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/genshen_h.hash"
        }
    },
    "anime2": {
        "bg": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime2_bg.pb",
            "path": resolve_relative_path('../.assets/models/style/anime2_bg.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime2_bg.hash"
        },
        "head": {
            "url": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime2_h.pb",
            "path": resolve_relative_path('../.assets/models/style/anime2_h.pb'),
            "hash": "https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/anime2_h.hash"
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
        #res = res[:, :, ::-1]  # Convert BGR to RGB
        return np.clip(res, 0, 255).astype(np.uint8)


def get_frame_processor() -> Any:
    global FRAME_PROCESSOR

    with THREAD_LOCK:
        if FRAME_PROCESSOR is None:
            model = get_options('model')
            model_name = MODEL_MAPPING.get(model)
            print(f"Loading style changer model: {model_name}")
            FRAME_PROCESSOR = ImageCartoonPipelineCustom(model=model)
    return FRAME_PROCESSOR


def clear_frame_processor() -> None:
    global FRAME_PROCESSOR
    if FRAME_PROCESSOR is not None:
        print("Deleting frame processor")
        del FRAME_PROCESSOR
        print("Frame processor deleted")
    FRAME_PROCESSOR = None


def get_options(key: Literal['model']) -> Any:
    global OPTIONS

    if OPTIONS is None:
        OPTIONS = \
            {
                'model': frame_processors_globals.style_changer_model,
                'target': frame_processors_globals.style_changer_target
            }
    return OPTIONS.get(key)


def set_options(key: Literal['model'], value: Any) -> None:
    global OPTIONS
    if not OPTIONS:
        OPTIONS = \
            {
                'model': MODEL_MAPPING[frame_processors_globals.style_changer_model],
                'target': frame_processors_globals.style_changer_target
            }
    if key == 'model' and OPTIONS.get(key) != value and FRAME_PROCESSOR:
        print("Clearing frame processor")
        clear_frame_processor()
        print("Frame processor cleared")
    OPTIONS[key] = value


# Register command-line arguments
def register_args(program: argparse.ArgumentParser) -> None:
    program.add_argument(
        '--style-changer-model',
        help=wording.get('help.style_changer_model'),
        default=config.get_str_value('style_changer_model', 'anime'),
        choices=list(MODEL_MAPPING.keys())
    )
    program.add_argument(
        '--style-changer-target',
        help=wording.get('help.style_changer_target'),
        default=config.get_str_value('style_changer_target', 'target'),
        choices=['source', 'target']
    )


# Apply command-line arguments
def apply_args(program: argparse.ArgumentParser) -> None:
    args = program.parse_args()
    facefusion.globals.style_changer_model = args.style_changer_model
    facefusion.globals.style_changer_target = args.style_changer_target


# Pre-check before processing
def pre_check() -> bool:
    model_name = facefusion.globals.style_changer_model
    if model_name not in MODEL_MAPPING:
        logger.error(f"Model '{model_name}' not found in available models.", "STYLE_CHANGER")
        return False
    print(f"Pre-check passed for model '{model_name}'.")
    download_path = resolve_relative_path('../.assets/models/style')
    model_urls = []
    for model_type, model in MODEL_MAPPING.items():
        print(f"Checking model '{model_type}'...")
        model_bg_url = model.get('bg').get('url')
        model_head_url = model.get('head').get('url')
        model_urls.append(model_bg_url)
        model_urls.append(model_head_url)
    conditional_download(download_path, model_urls)
    return True


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
    if get_options('target') == 'source':
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


def process_frames(source_paths: List[str], source_paths_2: List[str], queue_payloads: List[QueuePayload],
                   update_progress: UpdateProcess) -> None:
    if get_options('target') == 'source':
        print(f"Skipping processing for source target: {get_options('target')}")
        return
    for queue_payload in queue_payloads:
        target_vision_path = queue_payload['frame_path']
        target_vision_frame = read_image(target_vision_path)
        output_vision_frame = process_frame(
            {
                'target_vision_frame': target_vision_frame
            })
        write_image(target_vision_path, output_vision_frame)
        update_progress(target_vision_path)


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
    if get_options('target') == 'source':
        return
    target_vision_frame = read_static_image(target_path)
    output_vision_frame = process_frame(
        {
            'target_vision_frame': target_vision_frame
        })
    write_image(output_path, output_vision_frame)


def process_video(source_paths: List[str], source_paths_2: List[str], temp_frame_paths: List[str]) -> None:
    print("Processing video frames with style changer.")
    frame_processors.multi_process_frames(None, None, temp_frame_paths, process_frames)
