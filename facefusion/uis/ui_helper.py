import hashlib
import os
from typing import Optional

from facefusion import state_manager
from facefusion.filesystem import is_image, is_video


def convert_int_none(value: int) -> Optional[int]:
    if value == 'none':
        return None
    return value


def convert_str_none(value: str) -> Optional[str]:
    if value == 'none':
        return None
    return value


def suggest_output_path(output_directory_path: str, target_path: str) -> Optional[str]:
    if is_image(target_path) or is_video(target_path):
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        output_name = hashlib.sha1(str(state_manager.get_state()).encode('utf-8')).hexdigest()[:8]
        out_path = os.path.join(output_directory_path, f'{target_name}_{output_name}{target_extension}')
        return out_path
    return None
