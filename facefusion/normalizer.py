from typing import List, Optional
import os

from facefusion.filesystem import is_file, is_directory
from facefusion.typing import Padding, Fps
from modules.paths_internal import script_path


def normalize_output_path(source_paths: Optional[List[str]], target_path: Optional[str], output_path: Optional[str]) -> \
        Optional[str]:
    source_path = source_paths[0] if source_paths else None
    if output_path is None or not os.path.exists(output_path):
        output_path = os.path.join(script_path, "outputs", "facefusion")
        if not os.path.exists(output_path):
            print("Creating output directory.")
            os.makedirs(output_path)
    if is_file(source_path) and is_file(target_path) and is_directory(output_path):
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        output_path = os.path.join(output_path, source_name + '-' + target_name + target_extension)
    if is_file(target_path) and output_path:
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        output_name, output_extension = os.path.splitext(os.path.basename(output_path))
        output_directory_path = os.path.dirname(output_path)
        if is_directory(output_directory_path) and output_extension:
            output_path = os.path.join(output_directory_path, output_name + target_extension)
        else:
            output_path = None
    if output_path is not None and os.path.exists(output_path):
        out_idx = 1
        while True:
            output_path_without_extension = os.path.splitext(output_path)[0]
            output_path = output_path_without_extension + '_' + str(out_idx) + os.path.splitext(output_path)[1]
            out_idx += 1
            if not os.path.exists(output_path):
                break
    return output_path


def normalize_padding(padding: Optional[List[int]]) -> Optional[Padding]:
    if padding and len(padding) == 1:
        return tuple([padding[0], padding[0], padding[0], padding[0]])  # type: ignore[return-value]
    if padding and len(padding) == 2:
        return tuple([padding[0], padding[1], padding[0], padding[1]])  # type: ignore[return-value]
    if padding and len(padding) == 3:
        return tuple([padding[0], padding[1], padding[2], padding[1]])  # type: ignore[return-value]
    if padding and len(padding) == 4:
        return tuple(padding)  # type: ignore[return-value]
    return None


def normalize_fps(fps: Optional[float]) -> Optional[Fps]:
    if fps is not None:
        if fps < 1.0:
            return 1.0
        if fps > 60.0:
            return 60.0
        return fps
    return None
