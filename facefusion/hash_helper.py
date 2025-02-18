import os
import zlib
from typing import Optional

from facefusion.filesystem import is_file


def create_hash(content: bytes) -> str:
    return format(zlib.crc32(content), '08x')


def validate_hash(validate_path: str) -> bool:
    hash_path = get_hash_path(validate_path)

    if is_file(hash_path):
        with open(hash_path, 'r') as hash_file:
            hash_content = hash_file.read().strip()

        with open(validate_path, 'rb') as validate_file:
            validate_content = validate_file.read()
        hashed = create_hash(validate_content)
        if hashed != hash_content:
            print(f'Hash mismatch: {hash_path} != {hashed}')
        return hashed == hash_content
    print(f'Hash file not found: {hash_path}')
    return False


def get_hash_path(validate_path: str) -> Optional[str]:
    if is_file(validate_path):
        validate_directory_path, _ = os.path.split(validate_path)
        validate_file_name, _ = os.path.splitext(_)

        return os.path.join(validate_directory_path, validate_file_name + '.hash')
    return None


def create_file_hash(validate_path: str) -> Optional[str]:
    if is_file(validate_path):
        with open(validate_path, 'rb') as validate_file:
            validate_content = validate_file.read()
        hashed = create_hash(validate_content)
        hash_path = get_hash_path(validate_path)
        with open(hash_path, 'w') as hash_file:
            hash_file.write(hashed)
        return hashed
    return None
