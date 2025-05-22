import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import traceback
from typing import Dict, List, Tuple
from typing import Generic, Optional, TypeVar

import cv2
import numpy
import torch
from PIL import Image, ImageDraw
from cv2.typing import Size
from rich import print  # noqa: A004  Shadowing built-in 'print'
from torchvision.transforms.functional import to_pil_image
from ultralytics import YOLO, YOLOWorld

from facefusion import state_manager
from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.typing import DownloadSet, FaceLandmark68, FaceMaskRegion, Mask, ModelSet, Padding, \
    VisionFrame
from facefusion.workers.base_worker import BaseWorker
from modules.paths_internal import models_path

T = TypeVar("T", int, float)

logger = logging.getLogger(__name__)


@dataclass
class PredictOutput(Generic[T]):
    bboxes: list[list[T]] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    preview: Optional[Image.Image] = None


def create_mask_from_bbox(
        bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def ultralytics_predict(
        model_path: str | Path,
        image: Image.Image,
        confidence: float = 0.3,
        device: str = "",
        classes: str = "",
) -> PredictOutput[float]:
    from ultralytics import YOLO
    from modules import shared
    fuck_stupid_pickle_shit = False
    if not shared.cmd_opts.disable_safe_unpickle:
        shared.cmd_opts.disable_safe_unpickle = True
        fuck_stupid_pickle_shit = True

    model = YOLO(model_path)
    apply_classes(model, model_path, classes)
    pred = model(image, conf=confidence, device=device)

    if fuck_stupid_pickle_shit:
        shared.cmd_opts.disable_safe_unpickle = False

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()
    bboxes = bboxes.tolist()

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

    confidences = pred[0].boxes.conf.cpu().numpy().tolist()

    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
    )


def apply_classes(model: YOLO | YOLOWorld, model_path: str | Path, classes: str):
    if not classes or "-world" not in Path(model_path).stem:
        return
    parsed = [c.strip() for c in classes.split(",") if c.strip()]
    if parsed:
        model.set_classes(parsed)


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (W, H) of the original image
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]


class FaceMasker(BaseWorker):
    MODEL_SET: ModelSet = \
        {
            'face_occluder':
                {
                    'hashes':
                        {
                            'face_occluder':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/dfl_xseg.hash',
                                    'path': resolve_relative_path('../.assets/models/dfl_xseg.hash')
                                }
                        },
                    'sources':
                        {
                            'face_occluder':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/dfl_xseg.onnx',
                                    'path': resolve_relative_path('../.assets/models/dfl_xseg.onnx')
                                }
                        },
                    'size': (256, 256)
                },
            'face_parser':
                {
                    'hashes':
                        {
                            'face_parser':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.hash',
                                    'path': resolve_relative_path('../.assets/models/bisenet_resnet_34.hash')
                                }
                        },
                    'sources':
                        {
                            'face_parser':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx',
                                    'path': resolve_relative_path('../.assets/models/bisenet_resnet_34.onnx')
                                }
                        },
                    'size': (512, 512)
                }
        }
    FACE_MASK_REGIONS: Dict[FaceMaskRegion, int] = \
        {
            'skin': 1,
            'left-eyebrow': 2,
            'right-eyebrow': 3,
            'left-eye': 4,
            'right-eye': 5,
            'glasses': 6,
            'nose': 10,
            'mouth': 11,
            'upper-lip': 12,
            'lower-lip': 13
        }

    default_model = "face_occluder"
    multi_model = True
    preload = True
    model_key = None
    preferred_provider = 'cuda'

    def collect_model_downloads(self) -> Tuple[DownloadSet, DownloadSet]:
        model_hashes = \
            {
                'face_occluder': self.MODEL_SET.get('face_occluder').get('hashes').get('face_occluder'),
                'face_parser': self.MODEL_SET.get('face_parser').get('hashes').get('face_parser')
            }
        model_sources = \
            {
                'face_occluder': self.MODEL_SET.get('face_occluder').get('sources').get('face_occluder'),
                'face_parser': self.MODEL_SET.get('face_parser').get('sources').get('face_parser')
            }
        return model_hashes, model_sources

    @lru_cache(maxsize=None)
    def create_static_box_mask(self, crop_size: Size, face_mask_blur: float, face_mask_padding: Padding) -> Mask:
        blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
        blur_area = max(blur_amount // 2, 1)
        box_mask: Mask = numpy.ones(crop_size).astype(numpy.float32)
        box_mask[:max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
        box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)):, :] = 0
        box_mask[:, :max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
        box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)):] = 0
        if blur_amount > 0:
            box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
        return box_mask

    def create_occlusion_mask(self, crop_vision_frame: VisionFrame) -> Mask:
        model_size = self.MODEL_SET.get('face_occluder').get('size')
        prepare_vision_frame = cv2.resize(crop_vision_frame, model_size)
        prepare_vision_frame = numpy.expand_dims(prepare_vision_frame, axis=0).astype(numpy.float32) / 255
        prepare_vision_frame = prepare_vision_frame.transpose(0, 1, 2, 3)
        occlusion_mask = self.forward_occlude_face(prepare_vision_frame)
        occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(numpy.float32)
        occlusion_mask = cv2.resize(occlusion_mask, crop_vision_frame.shape[:2][::-1])
        occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return occlusion_mask

    def create_region_mask(self, crop_vision_frame: VisionFrame, face_mask_regions: List[FaceMaskRegion]) -> Mask:
        model_size = self.MODEL_SET.get('face_parser').get('size')
        prepare_vision_frame = cv2.resize(crop_vision_frame, model_size)
        prepare_vision_frame = prepare_vision_frame[:, :, ::-1].astype(numpy.float32) / 255
        prepare_vision_frame = numpy.subtract(prepare_vision_frame,
                                              numpy.array([0.485, 0.456, 0.406]).astype(numpy.float32))
        prepare_vision_frame = numpy.divide(prepare_vision_frame,
                                            numpy.array([0.229, 0.224, 0.225]).astype(numpy.float32))
        prepare_vision_frame = numpy.expand_dims(prepare_vision_frame, axis=0)
        prepare_vision_frame = prepare_vision_frame.transpose(0, 3, 1, 2)
        region_mask = self.forward_parse_face(prepare_vision_frame)
        region_mask = numpy.isin(region_mask.argmax(0),
                                 [self.FACE_MASK_REGIONS[region] for region in face_mask_regions])
        region_mask = cv2.resize(region_mask.astype(numpy.float32), crop_vision_frame.shape[:2][::-1])
        region_mask = (cv2.GaussianBlur(region_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return region_mask

    def create_custom_mask(self, crop_vision_frame: VisionFrame, face_landmark_5=None) -> Optional[Mask]:
        """Creates a mask using YOLO detection models"""
        model_path = state_manager.get_item('custom_yolo_model')

        adetailer_path = os.path.join(models_path, "adetailer")
        
        if model_path and not os.path.exists(model_path):
            # combine with the full path
            model_path = os.path.join(adetailer_path, model_path)
        
        if not model_path or not os.path.exists(model_path):
            logger.warning(f"Custom YOLO model path is not set or does not exist: {model_path}")
            return None

        confidence = state_manager.get_item('custom_yolo_confidence') or 0.5
        radius = state_manager.get_item('custom_yolo_radius') or 10

        try:
            # Convert image to numpy array if it's not already
            if isinstance(crop_vision_frame, Image.Image):
                img_array = numpy.array(crop_vision_frame)
            else:
                img_array = crop_vision_frame

            # Convert BGR to RGB if needed
            if img_array.shape[2] == 3:
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img_array

            # Create PIL image for processing
            pil_image = Image.fromarray(img_rgb)

            # Get bounding boxes and masks using our custom implementation
            device = "cuda" if torch.cuda.is_available() else "cpu"
            result = ultralytics_predict(model_path, pil_image, confidence, device=device)

            if not result.bboxes:
                logger.info("No objects detected by YOLO model")
                return None

            # Create mask from prediction results
            mask = numpy.zeros(crop_vision_frame.shape[:2], dtype=numpy.float32)

            # Apply masks if available
            if result.masks:
                for m in result.masks:
                    m_array = numpy.array(m)
                    mask = numpy.maximum(mask, m_array / 255.0)

            # Apply bounding boxes if no masks
            elif result.bboxes:
                for box in result.bboxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)

            # If face_landmark_5 is provided, prioritize detections near the face
            if face_landmark_5 is not None and mask.any():
                # Calculate the center of the face
                face_center = numpy.mean(face_landmark_5, axis=0)

                # Create a distance mask - closer to face gets higher priority
                h, w = mask.shape
                y, x = numpy.ogrid[:h, :w]
                face_center_x, face_center_y = face_center
                distance_mask = numpy.sqrt((x - face_center_x) ** 2 + (y - face_center_y) ** 2)

                # Normalize distance mask to [0, 1]
                distance_mask = distance_mask / distance_mask.max()

                # Weight the mask by distance (closer = higher value)
                mask = mask * (1 - distance_mask * 0.5)

            # Apply Gaussian blur for smooth edges
            if radius > 0:
                mask = cv2.GaussianBlur(mask, (0, 0), radius)

            # Normalize mask to [0, 1]
            if mask.max() > 0:
                mask = mask / mask.max()

            return mask

        except Exception as e:
            logger.error(f"Error in YOLO mask creation: {e}")
            print(f"Error in YOLO mask creation: {e}")
            traceback.print_exc()
            return None

    def prepare_image_for_model(self, image: Image.Image, model) -> torch.Tensor:
        """Prepare image for the model"""
        import torchvision.transforms as T

        # Convert PIL to tensor
        transform = T.Compose([
            T.ToTensor(),
        ])

        img_tensor = transform(image).unsqueeze(0)

        # Move to the same device as model
        try:
            # Check if the model has parameters
            if hasattr(model, 'parameters'):
                # Try to get the device from parameters
                try:
                    device = next(model.parameters()).device
                    img_tensor = img_tensor.to(device)
                except (StopIteration, RuntimeError):
                    # Fall back to CPU if parameter iteration fails
                    pass
            # If model doesn't have a parameters method or it's empty
            elif hasattr(model, 'device'):
                # Some models have a device attribute
                img_tensor = img_tensor.to(model.device)
        except Exception as e:
            logger.warning(f"Could not determine model device, using default: {e}")

        return img_tensor

    
    def create_preview(self, image, bboxes, confidences):
        """Create a preview image with bounding boxes"""
        import cv2

        # Convert PIL to cv2
        img_cv = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)

        # Draw boxes
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            conf = confidences[i] if i < len(confidences) else 0

            # Draw rectangle
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence
            cv2.putText(img_cv, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert back to PIL
        preview = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return preview

    def create_mouth_mask(self, face_landmark_68: FaceLandmark68) -> Mask:
        convex_hull = cv2.convexHull(face_landmark_68[numpy.r_[3:14, 31:36]].astype(numpy.int32))
        mouth_mask: Mask = numpy.zeros((512, 512)).astype(numpy.float32)
        mouth_mask = cv2.fillConvexPoly(mouth_mask, convex_hull, 1.0)  # type:ignore[call-overload]
        mouth_mask = cv2.erode(mouth_mask.clip(0, 1), numpy.ones((21, 3)))
        mouth_mask = cv2.GaussianBlur(mouth_mask, (0, 0), sigmaX=1, sigmaY=15)
        return mouth_mask

    def forward_occlude_face(self, prepare_vision_frame: VisionFrame) -> Mask:
        face_occluder = self.get_inference_pool().get('face_occluder')

        with conditional_thread_semaphore():
            occlusion_mask: Mask = face_occluder.run(None,
                                                     {
                                                         'input': prepare_vision_frame
                                                     })[0][0]

        return occlusion_mask

    def forward_parse_face(self, prepare_vision_frame: VisionFrame) -> Mask:
        face_parser = self.get_inference_pool().get('face_parser')

        with conditional_thread_semaphore():
            region_mask: Mask = face_parser.run(None,
                                                {
                                                    'input': prepare_vision_frame
                                                })[0][0]

        return region_mask
