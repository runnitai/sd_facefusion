from argparse import ArgumentParser
from typing import List, Tuple

import cv2
import numpy

from facefusion import config, logger, state_manager, wording
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.face_helper import warp_face_by_face_landmark_5
from facefusion.face_selector import sort_and_filter_faces
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import in_directory, same_file_extension
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.typing import FaceDebuggerInputs
from facefusion.program_helper import find_argument_group
from facefusion.typing import ApplyStateItem, Args, Face, ProcessMode, QueuePayload, VisionFrame
from facefusion.vision import read_image, read_static_image, write_image
from facefusion.workers.classes.face_masker import FaceMasker


class FaceDebugger(BaseProcessor):
    """
    Processor for debugging face-related attributes in images and videos.
    """
    MODEL_SET = {}
    model_key = None
    priority = 1000

    def register_args(self, program: ArgumentParser) -> None:
        group_processors = find_argument_group(program, "processors")
        if group_processors:
            group_processors.add_argument(
                "--face-debugger-items",
                help=wording.get("help.face_debugger_items").format(
                    choices=", ".join(["bounding-box", "face-landmark-5", "face-landmark-68", "face-mask"])
                ),
                default=config.get_str_list("processors.face_debugger_items", "face-landmark-5 face-mask"),
                choices=["bounding-box", "face-landmark-5", "face-landmark-68", "face-mask"],
                nargs="+",
                metavar="FACE_DEBUGGER_ITEMS",
            )

    def apply_args(self, args: Args, apply_state_item: ApplyStateItem) -> None:
        apply_state_item("face_debugger_items", args.get("face_debugger_items"))

    def pre_check(self) -> bool:
        return True

    def pre_process(self, mode: ProcessMode) -> bool:
        if mode == "output" and not in_directory(state_manager.get_item("output_path")):
            logger.error(wording.get("specify_image_or_video_output"), __name__)
            return False
        if mode == "output" and not same_file_extension(
                [state_manager.get_item("target_path"), state_manager.get_item("output_path")]
        ):
            logger.error(wording.get("match_target_and_output_extension"), __name__)
            return False
        return True

    def post_process(self) -> None:
        read_static_image.cache_clear()
        super().post_process()

    @staticmethod
    def debug_face(target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
        """
        Draws bounding boxes, landmarks, and mask outlines for debugging.
        """
        masker = FaceMasker()
        primary_color = (0, 0, 255)      # red
        secondary_color = (0, 255, 0)    # green
        tertiary_color = (255, 255, 0)   # cyan-ish or yellow-ish
        face_debugger_items = state_manager.get_item("face_debugger_items")
        face_mask_types = state_manager.get_item("face_mask_types")
        bounding_box = target_face.bounding_box.astype(numpy.int32)

        # Copy so we don't overwrite the original
        temp_vision_frame = temp_vision_frame.copy()

        # 1) Draw bounding box
        if "bounding-box" in face_debugger_items:
            x1, y1, x2, y2 = bounding_box
            cv2.rectangle(temp_vision_frame, (x1, y1), (x2, y2), primary_color, 2)

        # 2) Draw face mask(s)
        if "face-mask" in face_debugger_items:
            # Colors for each mask type
            mask_colors = {
                "box": (0, 255, 0),         # green
                "occlusion": (255, 0, 255), # magenta
                "region": (0, 165, 255),    # orange
            }

            # Warp face to a fixed size for consistent mask generation
            crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
                temp_vision_frame,
                target_face.landmark_set.get("5/68"),
                "arcface_128_v2",
                (512, 512)
            )
            inverse_matrix = cv2.invertAffineTransform(affine_matrix)
            temp_size = temp_vision_frame.shape[:2][::-1]

            # Collect masks
            masks = []
            if "box" in face_mask_types:
                box_mask = masker.create_static_box_mask(
                    crop_vision_frame.shape[:2][::-1],
                    0,  # zero blur for clarity
                    state_manager.get_item("face_mask_padding")
                )
                masks.append(("box", box_mask))

            if "occlusion" in face_mask_types:
                occlusion_mask = masker.create_occlusion_mask(crop_vision_frame)
                masks.append(("occlusion", occlusion_mask))

            if "region" in face_mask_types:
                region_mask = masker.create_region_mask(
                    crop_vision_frame,
                    state_manager.get_item("face_mask_regions")
                )
                masks.append(("region", region_mask))

            # For each mask, invert transform and draw contours
            for mask_name, c_mask in masks:
                c_mask_255 = (c_mask * 255).astype(numpy.uint8)
                # Warp mask back to original frame size
                inverse_mask = cv2.warpAffine(c_mask_255, inverse_matrix, temp_size)
                # Threshold to get contour
                _, inverse_mask = cv2.threshold(inverse_mask, 100, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(inverse_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                color = mask_colors.get(mask_name, (255, 255, 255))
                # Draw mask contour
                cv2.drawContours(temp_vision_frame, contours, -1, color, 2)

                # Label each contour
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.putText(
                        temp_vision_frame,
                        mask_name,
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )

        # 3) Draw face-landmark-5 or face-landmark-68 if desired
        if "face-landmark-5" in face_debugger_items:
            lm5 = target_face.landmark_set.get("5/68")
            if lm5 is not None:
                for (x, y) in lm5.astype(numpy.int32):
                    cv2.circle(temp_vision_frame, (x, y), 2, tertiary_color, -1)

        if "face-landmark-68" in face_debugger_items:
            lm68 = target_face.landmark_set.get("68")
            if lm68 is not None:
                for (x, y) in lm68.astype(numpy.int32):
                    cv2.circle(temp_vision_frame, (x, y), 1, tertiary_color, -1)

        return temp_vision_frame

    def process_frame(self, inputs: FaceDebuggerInputs) -> VisionFrame:
        reference_faces = inputs.get("reference_faces")
        target_vision_frame = inputs.get("target_vision_frame")
        many_faces = sort_and_filter_faces(get_many_faces([target_vision_frame]))

        face_selector_mode = state_manager.get_item("face_selector_mode")

        if face_selector_mode == "many":
            if many_faces:
                for target_face in many_faces:
                    target_vision_frame = self.debug_face(target_face, target_vision_frame)
        elif face_selector_mode == "one":
            target_face = get_one_face(many_faces)
            if target_face:
                target_vision_frame = self.debug_face(target_face, target_vision_frame)
        elif face_selector_mode == "reference":
            from facefusion.typing import Face
            from facefusion.face_selector import find_similar_faces
            distance_threshold = state_manager.get_item("reference_face_distance")
            for src_face_idx, ref_faces in reference_faces.items():
                if not ref_faces:
                    continue
                similar_faces = find_similar_faces(many_faces, ref_faces, distance_threshold)
                if similar_faces:
                    for similar_face in similar_faces:
                        target_vision_frame = self.debug_face(similar_face, target_vision_frame)

        return target_vision_frame

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        reference_faces, reference_faces_2 = (
            get_reference_faces() if "reference" in state_manager.get_item("face_selector_mode") else (None, None)
        )
        output_frames = []
        for queue_payload in queue_payloads:
            target_vision_path = queue_payload["frame_path"]
            target_vision_frame = read_image(target_vision_path)
            result_frame = self.process_frame(
                {
                    "reference_faces": reference_faces,
                    "reference_faces_2": reference_faces_2,
                    "target_vision_frame": target_vision_frame,
                }
            )
            write_image(target_vision_path, result_frame)
            output_frames.append((queue_payload["frame_number"], target_vision_path))
        return output_frames

    def process_image(self, target_path: str, output_path: str, reference_faces=None) -> None:
        if reference_faces is None:
            reference_faces = (
                get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else (None, None))
        target_vision_frame = read_static_image(target_path)
        result_frame = self.process_frame(
            {
                "reference_faces": reference_faces,
                "target_vision_frame": target_vision_frame,
            }
        )
        write_image(output_path, result_frame)
