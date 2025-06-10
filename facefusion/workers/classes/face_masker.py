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

    loaded_custom_model = None
    loaded_custom_model_path = None
    default_model = "face_occluder"
    multi_model = True
    preload = True
    model_key = None
    preferred_provider = 'cuda'

    # Cache for combined masks to replace lru_cache which has issues with unhashable types
    _combined_mask_cache = {}
    _combined_mask_cache_max_size = 100  # Limit cache size to prevent memory issues

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

    def create_combined_mask(self, face_mask_types, crop_size: Size, face_mask_blur: float, face_mask_padding: Padding,
                         face_mask_regions: List[FaceMaskRegion], crop_vision_frame, tmp_vision_frame,
                         face_landmark_5: FaceLandmark68, face=None):
        """
        Creates a combined mask from multiple mask types.
        
        Uses a custom caching mechanism to avoid issues with unhashable types in lru_cache.
        """
        # Debug - log if this is being called from debugger vs actual processing
        is_debug = False
        import traceback
        stack = traceback.extract_stack()
        for frame in stack:
            if 'face_debugger.py' in frame.filename:
                is_debug = True
                break
        
        context = "DEBUGGER" if is_debug else "PROCESSOR"

        # Create hashable key components
        try:
            if isinstance(face_mask_types, list):
                face_mask_types_key = tuple(sorted(face_mask_types))
            else:
                face_mask_types_key = face_mask_types

            if isinstance(face_mask_regions, list):
                face_mask_regions_key = tuple(sorted(face_mask_regions))
            else:
                face_mask_regions_key = face_mask_regions

            # Hash the frame shapes rather than the frames themselves
            crop_frame_shape = crop_vision_frame.shape if crop_vision_frame is not None else None
            tmp_frame_shape = tmp_vision_frame.shape if tmp_vision_frame is not None else None

            # Create a cache key - only use hashable components
            cache_key = (
                face_mask_types_key,
                crop_size,
                face_mask_blur,
                face_mask_padding,
                face_mask_regions_key,
                crop_frame_shape,
                tmp_frame_shape
            )

            # Check if we have this in cache
            if cache_key in self._combined_mask_cache:
                return self._combined_mask_cache[cache_key]

        except Exception as e:
            # If there's any error creating the cache key, just skip caching
            logger.warning(f"Error creating cache key: {e}")
            cache_key = None

        # Create the combined mask
        combined_mask = numpy.zeros(crop_size).astype(numpy.float32)

        # Skip processing if no mask types specified
        if not face_mask_types:
            return combined_mask

        # Prepare a dictionary to store individual masks
        masks = {}

        # First determine which mask types we need to create
        if isinstance(face_mask_types, tuple):
            face_mask_types_list = list(face_mask_types)
        else:
            face_mask_types_list = face_mask_types

        need_box = 'box' in face_mask_types_list
        need_occlusion = 'occlusion' in face_mask_types_list
        need_region = 'region' in face_mask_types_list
        need_custom = 'custom' in face_mask_types_list
        

        # Create each required mask type without blurring yet - we'll blur at the end
        # This helps prevent one mask from being overridden by another
        if need_box:
            # For box mask, we use zero blur for creation, will apply blur at the end
            masks['box'] = self.create_static_box_mask(crop_size, 0, face_mask_padding)

        if need_occlusion:
            masks['occlusion'] = self.create_occlusion_mask(crop_vision_frame)

        if need_region and face_mask_regions:
            # Ensure we're passing a list
            if isinstance(face_mask_regions, tuple):
                regions_list = list(face_mask_regions)
            else:
                regions_list = face_mask_regions

            masks['region'] = self.create_region_mask(crop_vision_frame, regions_list)

        if need_custom:
            # Pass both the crop frame and the full frame to the custom mask function
            custom_mask = self.create_custom_mask(
                crop_vision_frame=crop_vision_frame,
                face_landmark_or_face=face if face is not None else face_landmark_5,
                full_vision_frame=tmp_vision_frame
            )
            
            if custom_mask is not None and custom_mask.any():
                masks['custom'] = custom_mask

        # Combine all masks - each mask contributes separately to the final mask
        for mask_name, mask in masks.items():
            if mask is not None and mask.any():
                # Use maximum value at each pixel position
                combined_mask = numpy.maximum(combined_mask, mask)

        # Apply final blur to smooth edges if needed
        if combined_mask.any() and face_mask_blur > 0:
            blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
            if blur_amount > 0:
                combined_mask = cv2.GaussianBlur(combined_mask, (0, 0), blur_amount * 0.25)

        # Ensure mask values are in valid range
        combined_mask = combined_mask.clip(0, 1)


        # Cache the result if we have a valid cache key
        if cache_key is not None:
            # Manage cache size
            if len(self._combined_mask_cache) >= self._combined_mask_cache_max_size:
                # Remove a random item
                try:
                    self._combined_mask_cache.pop(next(iter(self._combined_mask_cache)))
                except Exception:
                    # If we can't remove an item, just clear the whole cache
                    self._combined_mask_cache.clear()

            # Add to cache
            self._combined_mask_cache[cache_key] = combined_mask

        return combined_mask

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

    def create_custom_mask(self, crop_vision_frame: VisionFrame, face_landmark_or_face=None, full_vision_frame=None) -> \
    Optional[Mask]:
        """Creates a mask using YOLO detection models
        
        Parameters
        ----------
        crop_vision_frame : VisionFrame
            The cropped face frame to create a mask for
        face_landmark_or_face : FaceLandmark5 or Face, optional
            Face landmark points or complete Face object to help with mask positioning
        full_vision_frame : VisionFrame, optional
            The full original frame for better object detection
        """
        model_path = state_manager.get_item('custom_yolo_model')
        adetailer_path = os.path.join(models_path, "adetailer")

        if model_path and not os.path.exists(model_path):
            # combine with the full path
            model_path = os.path.join(adetailer_path, model_path)

        if not model_path or not os.path.exists(model_path):
            logger.warning(f"Custom YOLO model path is not set or does not exist: {model_path}")
            self.loaded_custom_model = None
            self.loaded_custom_model_path = None
            return None

        confidence = state_manager.get_item('custom_yolo_confidence') or 0.5
        radius = state_manager.get_item('custom_yolo_radius') or 10
        
        # Debug - log if this is being called from debugger vs actual processing
        is_debug = False
        import traceback
        stack = traceback.extract_stack()
        for frame in stack:
            if 'face_debugger.py' in frame.filename:
                is_debug = True
                break
        
        context = "DEBUGGER" if is_debug else "PROCESSOR"
        logger.debug(f"[{context}] Creating custom mask - full_frame: {full_vision_frame is not None}")

        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(resolve_relative_path('../.debug'), 'face_masker')
        os.makedirs(debug_dir, exist_ok=True)

        try:
            # Extract face information
            face_obj = None
            face_landmarks = None
            face_landmark_5 = None

            if face_landmark_or_face is not None:
                if hasattr(face_landmark_or_face, 'landmark_set'):
                    # This is a Face object
                    face_obj = face_landmark_or_face
                    if '5/68' in face_obj.landmark_set:
                        face_landmark_5 = face_obj.landmark_set['5/68']
                    elif '5' in face_obj.landmark_set:
                        face_landmark_5 = face_obj.landmark_set['5']
                    
                    face_landmarks = face_landmark_5
                else:
                    # This is a landmark array
                    face_landmarks = face_landmark_or_face
                    face_landmark_5 = face_landmark_or_face
            
            # If we don't have landmarks, we can't proceed with accurate transformations
            if face_landmark_5 is None or len(face_landmark_5) < 5:
                logger.warning("No valid face landmarks found for mask transformation")
                return None
            
            # Determine which frame to use for object detection
            detection_frame = full_vision_frame if full_vision_frame is not None else crop_vision_frame

            # Convert detection frame to PIL image for the model
            if isinstance(detection_frame, Image.Image):
                pil_image = detection_frame
            else:
                # Convert numpy array to PIL
                if detection_frame.shape[2] == 3:
                    rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = detection_frame
                pil_image = Image.fromarray(rgb_frame)

            # Run the YOLO model for detection
            device = "cuda" if torch.cuda.is_available() else "cpu"
            result = self.ultralytics_predict(model_path, pil_image, confidence, device=device)
            
            # Debug detection results
            if result.bboxes:
                logger.debug(f"[{context}] Detected {len(result.bboxes)} bounding boxes")
                for i, box in enumerate(result.bboxes):
                    conf = result.confidences[i] if i < len(result.confidences) else 0
                    logger.debug(f"[{context}] Box {i}: {[int(b) for b in box]} - Confidence: {conf:.2f}")
            if result.masks:
                logger.debug(f"[{context}] Detected {len(result.masks)} masks")

            if not result.bboxes and not result.masks:
                logger.info("No objects detected by YOLO model")
                return None

            # Create base mask for the crop frame
            crop_h, crop_w = crop_vision_frame.shape[:2]
            crop_size = (crop_w, crop_h)
            mask = numpy.zeros((crop_h, crop_w), dtype=numpy.float32)
            
            # If we're using the full frame for detection, we need to transform the masks
            if full_vision_frame is not None:
                # Important: We need to use the same affine transform that was used to create the crop_vision_frame
                # This ensures proper alignment between the detected objects and the crop frame
                
                # Get the warp template used in the face swapper
                warp_template = "arcface_128_v2"  # This should match what's used in face_swapper.py
                
                # Estimate the affine matrix - this should be the same matrix used to create crop_vision_frame
                from facefusion.face_helper import estimate_matrix_by_face_landmark_5
                
                try:
                    # Calculate the affine matrix from face landmarks to crop frame
                    affine_matrix = estimate_matrix_by_face_landmark_5(
                        face_landmark_5, 
                        warp_template, 
                        crop_size
                    )
                    
                    logger.debug(f"[{context}] Affine matrix: {affine_matrix}")
                    
                    # Save a debug image of the full frame with bounding boxes for visual reference
                    if is_debug and result.preview is not None:
                        debug_preview_path = os.path.join(debug_dir, "detection_preview.png")
                        result.preview.save(debug_preview_path)
                        logger.debug(f"[{context}] Saved detection preview to {debug_preview_path}")
                    
                    # Also save the cropped frame for reference
                    if is_debug:
                        debug_crop_path = os.path.join(debug_dir, "crop_frame.png")
                        cv2.imwrite(debug_crop_path, cv2.cvtColor(crop_vision_frame, cv2.COLOR_BGR2RGB))
                        logger.debug(f"[{context}] Saved crop frame to {debug_crop_path}")
                    
                    # Process detected masks using this transformation matrix
                    if result.masks:
                        for j, m in enumerate(result.masks):
                            # Convert PIL mask to numpy array (0-255)
                            m_array = numpy.array(m)
                            logger.debug(f"[{context}] Mask {j} size: {m_array.shape}, sum: {m_array.sum()}, max: {m_array.max()}")
                            
                            # Resize the mask to match the full frame if needed
                            full_h, full_w = full_vision_frame.shape[:2]
                            if m_array.shape[:2] != (full_h, full_w):
                                m_array = cv2.resize(m_array, (full_w, full_h))
                                logger.debug(f"[{context}] Resized mask {j} to {m_array.shape}")
                            
                            # Save original mask with the full frame for visual context
                            if is_debug:
                                # Create a colored overlay of the mask on the full frame
                                overlay = full_vision_frame.copy()
                                mask_colored = cv2.merge([
                                    numpy.zeros_like(m_array),  # Blue channel
                                    m_array,                   # Green channel
                                    numpy.zeros_like(m_array)  # Red channel
                                ])
                                # Apply the overlay with 50% opacity
                                debug_orig_vis = cv2.addWeighted(
                                    overlay, 1.0,
                                    mask_colored, 0.5,
                                    0
                                )
                                debug_mask_path = os.path.join(debug_dir, f"mask_{j}_orig_vis.png")
                                cv2.imwrite(debug_mask_path, debug_orig_vis)
                                logger.debug(f"[{context}] Saved original mask visualization to {debug_mask_path}")
                            
                            # Apply the same affine transformation to the mask as was applied to the face
                            # This ensures perfect alignment with the crop_vision_frame
                            m_warped = cv2.warpAffine(
                                m_array, 
                                affine_matrix, 
                                crop_size,
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=0
                            )
                            
                            logger.debug(f"[{context}] Warped mask {j} size: {m_warped.shape}, sum: {m_warped.sum()}, max: {m_warped.max()}")
                            
                            # Save warped mask with cropped frame for context
                            if is_debug:
                                # Create a colored overlay of the warped mask on the crop frame
                                crop_overlay = crop_vision_frame.copy()
                                warped_colored = cv2.merge([
                                    numpy.zeros_like(m_warped),  # Blue channel
                                    m_warped,                    # Green channel
                                    numpy.zeros_like(m_warped)   # Red channel
                                ])
                                # Apply the overlay with 50% opacity
                                debug_warped_vis = cv2.addWeighted(
                                    crop_overlay, 1.0,
                                    warped_colored, 0.5,
                                    0
                                )
                                debug_mask_path = os.path.join(debug_dir, f"mask_{j}_warped_vis.png")
                                cv2.imwrite(debug_mask_path, debug_warped_vis)
                                logger.debug(f"[{context}] Saved warped mask visualization to {debug_mask_path}")
                            
                            # Instead of just using bounding boxes, try to preserve the actual mask shape
                            # This should help create more precise masks than just rectangles
                            mask = numpy.maximum(mask, m_warped / 255.0)
                            logger.debug(f"[{context}] After adding mask {j}: sum={mask.sum()}, max={mask.max()}")
                    
                    elif result.bboxes:
                        # If no masks available, use bounding boxes
                        full_h, full_w = full_vision_frame.shape[:2]
                        
                        # Create a blank mask for the full frame
                        full_mask = numpy.zeros((full_h, full_w), dtype=numpy.uint8)
                        
                        # Draw all bounding boxes on the mask - use elliptical shapes for more natural masks
                        for j, box in enumerate(result.bboxes):
                            x1, y1, x2, y2 = [int(b) for b in box]
                            logger.debug(f"[{context}] Box {j}: ({x1}, {y1}) - ({x2}, {y2})")
                            
                            # Ensure coordinates are within bounds
                            x1 = max(0, min(full_w - 1, x1))
                            y1 = max(0, min(full_h - 1, y1))
                            x2 = max(0, min(full_w - 1, x2))
                            y2 = max(0, min(full_h - 1, y2))
                            
                            # Draw filled ellipse instead of rectangle for more natural shape
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
                            cv2.ellipse(full_mask, center, axes, 0, 0, 360, 255, -1)
                            logger.debug(f"[{context}] Drew ellipse at center {center} with axes {axes}")
                        
                        # Save the full mask visualization
                        if is_debug:
                            # Create a colored overlay of the mask on the full frame
                            overlay = full_vision_frame.copy()
                            mask_colored = cv2.merge([
                                numpy.zeros_like(full_mask),  # Blue channel
                                full_mask,                    # Green channel
                                numpy.zeros_like(full_mask)   # Red channel
                            ])
                            # Apply the overlay with 50% opacity
                            debug_full_vis = cv2.addWeighted(
                                overlay, 1.0,
                                mask_colored, 0.5,
                                0
                            )
                            debug_mask_path = os.path.join(debug_dir, "box_mask_full_vis.png")
                            cv2.imwrite(debug_mask_path, debug_full_vis)
                            logger.debug(f"[{context}] Saved full box mask visualization to {debug_mask_path}")
                        
                        # Apply the same affine transformation to the mask
                        full_mask_warped = cv2.warpAffine(
                            full_mask, 
                            affine_matrix, 
                            crop_size,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=0
                        )
                        
                        # Save the warped mask visualization
                        if is_debug:
                            # Create a colored overlay of the warped mask on the crop frame
                            crop_overlay = crop_vision_frame.copy()
                            warped_colored = cv2.merge([
                                numpy.zeros_like(full_mask_warped),  # Blue channel
                                full_mask_warped,                    # Green channel
                                numpy.zeros_like(full_mask_warped)   # Red channel
                            ])
                            # Apply the overlay with 50% opacity
                            debug_warped_vis = cv2.addWeighted(
                                crop_overlay, 1.0,
                                warped_colored, 0.5,
                                0
                            )
                            debug_mask_path = os.path.join(debug_dir, "box_mask_warped_vis.png")
                            cv2.imwrite(debug_mask_path, debug_warped_vis)
                            logger.debug(f"[{context}] Saved warped box mask visualization to {debug_mask_path}")
                        
                        # Add to the mask
                        mask = numpy.maximum(mask, full_mask_warped / 255.0)
                        logger.debug(f"[{context}] Added box mask with sum: {full_mask_warped.sum()}, max: {full_mask_warped.max()}")
                
                except Exception as e:
                    logger.error(f"Error applying affine transformation: {e}")
                    traceback.print_exc()
                    # Fall back to using the crop frame directly
            
            # If we're using the crop frame directly, or if the transformation failed
            if mask.sum() == 0:
                logger.debug(f"[{context}] Using crop frame directly for detection")
                # Apply masks if available
                if result.masks:
                    for j, m in enumerate(result.masks):
                        m_array = numpy.array(m)
                        # Resize if needed
                        if m_array.shape[:2] != (crop_h, crop_w):
                            m_array = cv2.resize(m_array, (crop_w, crop_h))
                        
                        # Save mask visualization for debugging
                        if is_debug:
                            # Create a colored overlay of the mask on the crop frame
                            crop_overlay = crop_vision_frame.copy()
                            mask_colored = cv2.merge([
                                numpy.zeros_like(m_array),  # Blue channel
                                m_array,                    # Green channel
                                numpy.zeros_like(m_array)   # Red channel
                            ])
                            # Apply the overlay with 50% opacity
                            debug_mask_vis = cv2.addWeighted(
                                crop_overlay, 1.0,
                                mask_colored, 0.5,
                                0
                            )
                            debug_mask_path = os.path.join(debug_dir, f"direct_mask_{j}_vis.png")
                            cv2.imwrite(debug_mask_path, debug_mask_vis)
                            logger.debug(f"[{context}] Saved direct mask visualization to {debug_mask_path}")
                        
                        mask = numpy.maximum(mask, m_array / 255.0)
                        logger.debug(f"[{context}] Added direct mask {j} with sum: {m_array.sum()}, max: {m_array.max()}")

                # Apply bounding boxes if no masks
                elif result.bboxes:
                    for j, box in enumerate(result.bboxes):
                        x1, y1, x2, y2 = [int(b) for b in box]
                        # Ensure coordinates are within bounds
                        x1 = max(0, min(crop_w - 1, x1))
                        y1 = max(0, min(crop_h - 1, y1))
                        x2 = max(0, min(crop_w - 1, x2))
                        y2 = max(0, min(crop_h - 1, y2))
                        
                        logger.debug(f"[{context}] Processing direct box {j}: ({x1}, {y1}) - ({x2}, {y2})")
                        
                        # Use ellipse for more natural shape
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
                        
                        # Create a temporary mask for this box
                        box_mask = numpy.zeros((crop_h, crop_w), dtype=numpy.float32)
                        cv2.ellipse(box_mask, center, axes, 0, 0, 360, 1.0, -1)
                        
                        # Save box mask visualization for debugging
                        if is_debug:
                            # Create a colored overlay of the box mask on the crop frame
                            crop_overlay = crop_vision_frame.copy()
                            box_colored = cv2.merge([
                                numpy.zeros_like(box_mask),  # Blue channel
                                (box_mask * 255).astype(numpy.uint8),  # Green channel
                                numpy.zeros_like(box_mask)   # Red channel
                            ])
                            # Apply the overlay with 50% opacity
                            debug_box_vis = cv2.addWeighted(
                                crop_overlay, 1.0,
                                box_colored, 0.5,
                                0
                            )
                            debug_box_path = os.path.join(debug_dir, f"direct_box_{j}_vis.png")
                            cv2.imwrite(debug_box_path, debug_box_vis)
                            logger.debug(f"[{context}] Saved direct box visualization to {debug_box_path}")
                        
                        mask = numpy.maximum(mask, box_mask)
                        logger.debug(f"[{context}] Added direct box mask {j} with sum: {box_mask.sum()}, max: {box_mask.max()}")

            # If face landmarks are available, weight the mask by distance to face center
            # This helps prioritize detected objects closer to the face
            if face_landmarks is not None and mask.any():
                # Calculate the center of the face in the crop frame
                face_center = numpy.mean(face_landmarks, axis=0)

                # Create a distance mask - closer to face gets higher priority
                h, w = mask.shape
                y, x = numpy.ogrid[:h, :w]
                face_center_x, face_center_y = face_center
                distance_mask = numpy.sqrt((x - face_center_x) ** 2 + (y - face_center_y) ** 2)

                # Normalize distance mask to [0, 1]
                max_dist = numpy.sqrt(w**2 + h**2)
                distance_mask = distance_mask / max_dist
                
                # Save the distance mask for debugging
                if is_debug and mask.any():
                    # Visualize the distance mask
                    distance_vis = (1 - distance_mask * 0.3) * 255
                    distance_vis_path = os.path.join(debug_dir, "distance_mask.png")
                    cv2.imwrite(distance_vis_path, distance_vis.astype(numpy.uint8))
                    logger.debug(f"[{context}] Saved distance mask to {distance_vis_path}")
                
                # Weight the mask by distance (closer = higher value)
                # Less aggressive falloff for better coverage
                mask_before = mask.copy()
                mask = mask * (1 - distance_mask * 0.3)
                
                # Save the weighted mask for debugging
                if is_debug and mask.any():
                    # Create before/after comparison
                    before_after = numpy.hstack([
                        (mask_before * 255).astype(numpy.uint8),
                        (mask * 255).astype(numpy.uint8)
                    ])
                    before_after_path = os.path.join(debug_dir, "mask_weighting.png")
                    cv2.imwrite(before_after_path, before_after)
                    logger.debug(f"[{context}] Saved before/after distance weighting to {before_after_path}")

            # Apply Gaussian blur for smooth edges
            # Store pre-blur mask for comparison
            pre_blur_mask = None
            if mask.any() and radius > 0:
                pre_blur_mask = mask.copy()
                
                # Find edges for targeted blurring
                edges = cv2.Canny(
                    (mask * 255).astype(numpy.uint8), 
                    threshold1=50, 
                    threshold2=150
                )
                
                # Dilate the edges to create a region to blur
                kernel = numpy.ones((3, 3), numpy.uint8)
                edge_region = cv2.dilate(edges, kernel, iterations=2)
                
                # Create a blurred version of the entire mask
                blurred_mask = cv2.GaussianBlur(mask, (0, 0), radius)
                
                # Only apply blur to the edge regions
                edge_region = edge_region.astype(numpy.float32) / 255.0
                mask = mask * (1 - edge_region) + blurred_mask * edge_region
                
                # Save edge-aware blur visualization
                if is_debug:
                    # Create comparison of pre-blur, edges, and post-blur
                    edge_vis = (edge_region * 255).astype(numpy.uint8)
                    blur_comparison = numpy.hstack([
                        (pre_blur_mask * 255).astype(numpy.uint8),
                        edge_vis,
                        (mask * 255).astype(numpy.uint8)
                    ])
                    blur_path = os.path.join(debug_dir, "edge_blur_comparison.png")
                    cv2.imwrite(blur_path, blur_comparison)
                    logger.debug(f"[{context}] Saved edge-aware blur comparison to {blur_path}")
            
            # Normalize mask to [0, 1]
            if mask.max() > 0:
                mask = mask / mask.max()

            # Save final mask visualization
            if is_debug and mask.any():
                # Apply mask to crop frame for final visualization
                crop_result = crop_vision_frame.copy()
                # Use alpha blending to show masked area
                alpha_mask = numpy.stack([mask] * 3, axis=2)
                mask_highlight = crop_vision_frame.copy() * 0.7 + numpy.array([0, 100, 0], dtype=numpy.uint8) * alpha_mask
                final_vis = cv2.addWeighted(crop_result, 0.7, mask_highlight.astype(numpy.uint8), 0.3, 0)
                final_vis_path = os.path.join(debug_dir, "final_mask_vis.png")
                cv2.imwrite(final_vis_path, final_vis)
                logger.debug(f"[{context}] Saved final mask visualization to {final_vis_path}")

            # Log mask properties for debugging
            if mask.any():
                logger.debug(f"Custom mask created - sum: {mask.sum()}, max: {mask.max()}, size: {mask.shape}")
                # Calculate what percentage of the frame is covered by the mask
                coverage = (mask.sum() / (mask.shape[0] * mask.shape[1])) * 100
                logger.debug(f"[{context}] Mask coverage: {coverage:.2f}% of frame")
            
            return mask

        except Exception as e:
            logger.error(f"Error in YOLO mask creation: {e}")
            print(f"Error in YOLO mask creation: {e}")
            traceback.print_exc()
            return None

    def ultralytics_predict(
            self,
            model_path: str,
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

        if not self.loaded_custom_model or model_path != self.loaded_custom_model_path:
            model = YOLO(model_path)
            self.loaded_custom_model_path = model_path
            self.loaded_custom_model = model
        else:
            model = self.loaded_custom_model
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
        logger.debug(f"We have a total of {len(masks)} masks and {len(bboxes)} bboxes.")
        return PredictOutput(
            bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
        )

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
        """Create a detailed preview image with bounding boxes, confidence scores, and additional information"""
        import cv2

        # Convert PIL to cv2
        img_cv = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)

        # Create a copy for drawing
        preview_img = img_cv.copy()

        # Define colors for different confidence levels
        colors = {
            'high': (0, 255, 0),  # Green for high confidence
            'medium': (0, 255, 255),  # Yellow for medium confidence
            'low': (0, 0, 255)  # Red for low confidence
        }

        # Draw boxes
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            conf = confidences[i] if i < len(confidences) else 0

            # Determine color based on confidence
            if conf > 0.7:
                color = colors['high']
            elif conf > 0.4:
                color = colors['medium']
            else:
                color = colors['low']

            # Draw rectangle with thickness based on confidence
            thickness = max(1, int(conf * 5))
            cv2.rectangle(preview_img, (x1, y1), (x2, y2), color, thickness)

            # Create label with confidence percentage
            label = f"{int(conf * 100)}%"

            # Get text size for background rectangle
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_w, text_h = text_size

            # Draw background rectangle for text
            cv2.rectangle(
                preview_img,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 10, y1),
                color,
                -1
            )

            # Draw text
            cv2.putText(
                preview_img,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            # Optionally draw object size
            object_width = x2 - x1
            object_height = y2 - y1
            size_label = f"{object_width}x{object_height}"

            cv2.putText(
                preview_img,
                size_label,
                (x1 + 5, y2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

        # Add summary information at the top
        if bboxes:
            info_text = f"Detected: {len(bboxes)} objects"
            cv2.putText(
                preview_img,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        else:
            # If no detections, indicate this prominently
            cv2.putText(
                preview_img,
                "No objects detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Add YOLO model name if available
        model_path = state_manager.get_item('custom_yolo_model')
        if model_path:
            model_name = os.path.basename(model_path)
            cv2.putText(
                preview_img,
                f"Model: {model_name}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            # Add confidence threshold
            confidence = state_manager.get_item('custom_yolo_confidence') or 0.5
            cv2.putText(
                preview_img,
                f"Threshold: {confidence:.2f}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # Convert back to PIL
        preview = Image.fromarray(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB))
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
