from argparse import ArgumentParser
from typing import List, Tuple, Union

import cv2
import torch

from facefusion import logger, wording, state_manager, process_manager
from facefusion.filesystem import resolve_relative_path, in_directory, is_image, is_video, same_file_extension
from facefusion.jobs import job_store
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.typing import VisionFrame
from facefusion.style_network import Stylization, ReshapeTool, TransformerNet
from facefusion.typing import QueuePayload, ApplyStateItem, Args, ProcessMode
from facefusion.vision import read_image, read_static_image, write_image


class StyleTransfer(BaseProcessor):
    MODEL_SET = {
        'style_transfer': {
            'hashes': {
                'style_transfer': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/style_net-TIP-final.hash',
                    'path': resolve_relative_path('../.assets/models/style/style_net-TIP-final.hash')
                }
            },
            'sources': {
                'style_transfer': {
                    'url': 'https://github.com/runnitai/sd_facefusion/releases/download/1.0.0/style_net-TIP-final.pth',
                    'path': resolve_relative_path('../.assets/models/style/style_net-TIP-final.pth')
                }
            }
        }
    }

    model_key = 'style_transfer_model'
    is_face_processor: bool = False
    style_input_path: Union[str, List[str]] = None
    default_model = 'style_transfer'
    priority = 1000
    frame_processor = None
    style_input = None
    reshape_tool = None

    def register_args(self, program: ArgumentParser) -> None:
        group_processors = program.add_argument_group('processors')
        group_processors.add_argument('--style-transfer-model',
                                      help=wording.get('help.style_transfer_model'),
                                      default='style_transfer',
                                      choices=['style_transfer'])
        job_store.register_step_keys(["style_transfer_model", "style_transfer_images"])

    def apply_args(self, args: Args, apply_state_item: ApplyStateItem) -> None:
        apply_state_item('style_transfer_model', args.get('style_transfer_model'))

    def pre_process(self, mode: ProcessMode) -> bool:
        if mode in ['output', 'preview'] and not (is_image(state_manager.get_item('target_path')) or
                                                  is_video(state_manager.get_item('target_path'))):
            logger.error(wording.get('choose_image_or_video_target') + wording.get('exclamation_mark'), __name__)
            return False

        if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
            logger.error(wording.get('specify_image_or_video_output') + wording.get('exclamation_mark'), __name__)
            return False

        if mode == 'output' and not same_file_extension([
            state_manager.get_item('target_path'), state_manager.get_item('output_path')]):
            logger.error(wording.get('match_target_and_output_extension') + wording.get('exclamation_mark'), __name__)
            return False

        style_path = state_manager.get_item('style_transfer_images')
        if not style_path:
            logger.error("No style_image provided.", __name__)
            return False

        self.get_frame_processor()
        return True

    def process_frame(self, inputs: dict) -> VisionFrame:
        target_vision_frame = inputs.get('target_vision_frame')
        self.load_processor_global([target_vision_frame])
        return self.run_style_transfer(target_vision_frame)

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        processed_frames = []
        for queue_payload in process_manager.manage(queue_payloads):
            target_vision_path = queue_payload['frame_path']
            is_preview = queue_payload.get('is_preview', False)
            target_vision_frame = read_static_image(target_vision_path)
            output_vision_frame = self.process_frame(
                {'target_vision_frame': target_vision_frame, 'is_preview': is_preview})
            write_image(target_vision_path, output_vision_frame)
            processed_frames.append((queue_payload['frame_number'], target_vision_path))
        return processed_frames

    def process_image(self, target_path: str, output_path: str) -> None:
        target_vision_frame = read_static_image(target_path)
        #self.load_processor_global([target_path])
        output_vision_frame = self.process_frame({'target_vision_frame': target_vision_frame})
        write_image(output_path, output_vision_frame)

    def get_frame_processor(self):
        if self.frame_processor is None or state_manager.get_item('style_transfer_images') != self.style_input_path:
            checkpoint_path = self.get_model_options()['sources']['style_transfer']['path']
            cuda = torch.cuda.is_available()
            frame_processor = TransformerNet()
            frame_processor.load_state_dict(torch.load(checkpoint_path), strict=False)
            if cuda:
                frame_processor.cuda()

            for param in frame_processor.parameters():
                param.requires_grad = False
            styles = state_manager.get_item('style_transfer_images')
            style_images = []
            for style in styles:
                style_image = cv2.imread(style)
                style_images.append(style_image)

            framework = Stylization(checkpoint_path, cuda, style_num=len(style_images))
            framework.clean()
            framework.prepare_styles(style_images)

            self.frame_processor = framework
            self.style_input_path = styles

        if not self.reshape_tool:
            self.reshape_tool = ReshapeTool()

        return self.frame_processor

    def load_processor_global(self, target_frames: List[VisionFrame]):
        processor = self.get_frame_processor()
        if not processor.computed:
            processor.prepare_global(target_frames)
        return processor

    def run_style_transfer(self, frame: VisionFrame) -> VisionFrame:
        H, W, C = frame.shape
        reshape = self.reshape_tool or ReshapeTool()
        new_input_frame = reshape.process(frame)
        styled_input_frame = self.frame_processor.transfer(new_input_frame)
        styled_input_frame = styled_input_frame[64:64 + H, 64:64 + W, :]
        return styled_input_frame
