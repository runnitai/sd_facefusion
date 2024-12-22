import json
from typing import Optional

import numpy as np

from facefusion.filesystem import is_file
from facefusion.typing import Face


class FaceEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, Face):
            return {
                'bounding_box': obj.bounding_box.tolist(),
                'score_set': obj.score_set,
                'landmark_set': {k: v.tolist() for k, v in obj.landmark_set.items()},
                'angle': obj.angle,
                'embedding': obj.embedding.tolist(),
                'normed_embedding': obj.normed_embedding.tolist(),
                'gender': obj.gender,
                'age': {
                    'start': obj.age.start,
                    'stop': obj.age.stop,
                    'step': obj.age.step
                } if isinstance(obj.age, range) else obj.age,
                'race': obj.race
            }
        if isinstance(obj, range):
            return {
                '__range__': True,
                'start': obj.start,
                'stop': obj.stop,
                'step': obj.step
            }
        return super().default(obj)


def face_decoder(dct):
    if '__range__' in dct:
        return range(dct['start'], dct['stop'], dct['step'])
    if 'bounding_box' in dct and 'embedding' in dct and 'landmark_set' in dct:
        return Face(
            bounding_box=np.array(dct['bounding_box'], dtype=np.float32),
            score_set=dct['score_set'],
            landmark_set={k: np.array(v, dtype=np.float32) for k, v in dct['landmark_set'].items()},
            angle=dct['angle'],
            embedding=np.array(dct['embedding'], dtype=np.float32),
            normed_embedding=np.array(dct['normed_embedding'], dtype=np.float32),
            gender=dct['gender'],
            age=dct['age'] if isinstance(dct['age'], range) else range(dct['age']['start'], dct['age']['stop'],
                                                                       dct['age']['step']),
            race=dct['race']
        )
    return dct


def read_json(json_path: str) -> Optional[dict]:
    if is_file(json_path):
        try:
            with open(json_path, 'r') as json_file:
                return json.load(json_file, object_hook=face_decoder)
        except json.JSONDecodeError as e:
            print(f"JSON decode error for {json_path}: {e}")
    return None


def dicts_are_equal(dict1: dict, dict2: dict) -> bool:
    if dict1.keys() != dict2.keys():
        print("Keys mismatch:", dict1.keys(), "vs", dict2.keys())
        return False
    for key in dict1:
        if isinstance(dict1[key], np.ndarray) and isinstance(dict2[key], np.ndarray):
            if not np.array_equal(dict1[key], dict2[key]):
                print(f"Mismatch in ndarray at key '{key}': {dict1[key]} vs {dict2[key]}")
                return False
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not dicts_are_equal(dict1[key], dict2[key]):
                print(f"Mismatch in nested dict at key '{key}'")
                return False
        elif dict1[key] != dict2[key]:
            print(f"Mismatch at key '{key}': {dict1[key]} vs {dict2[key]}")
            return False
    return True


def write_json(json_path: str, content: dict) -> bool:
    try:
        with open(json_path, 'w') as json_file:
            json.dump(content, json_file, indent=4, cls=FaceEncoder)

        read_content = read_json(json_path)
        if read_content is None:
            print(f"Failed to read back JSON from {json_path}.")
            return False

        if not dicts_are_equal(read_content, content):
            print(f"Conversion mismatch in {json_path}!")

        return is_file(json_path)
    except Exception as e:
        print(f"Error writing JSON to {json_path}: {e}")
        return False
