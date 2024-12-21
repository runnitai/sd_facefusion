import json
from typing import Optional, Any
import numpy as np
from facefusion.filesystem import is_file
from facefusion.typing import Content, Face


# Define serialization and deserialization hooks
def custom_serializer(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        # Convert numpy array to flattened bit string for compactness
        flat_array = obj.flatten()
        return {
            "__type__": "ndarray",
            "shape": obj.shape,
            "data": flat_array.tolist()
        }
    if isinstance(obj, np.float32):  # Handle numpy.float32
        return {"__type__": "float32", "data": float(obj)}
    if isinstance(obj, range):  # Serialize range as a dictionary
        return {"__type__": "range", "start": obj.start, "stop": obj.stop, "step": obj.step}
    if isinstance(obj, Face):  # Assuming Face is a defined class
        return {
            "__type__": "Face",
            "bounding_box": obj.bounding_box.tolist(),
            "score_set": obj.score_set,
            "landmark_set": {k: v.tolist() for k, v in obj.landmark_set.items()},
            "angle": obj.angle,
            "embedding": obj.embedding.tolist(),
            "normed_embedding": obj.normed_embedding.tolist(),
            "gender": obj.gender,
            "age": {"__type__": "range", "start": obj.age.start, "stop": obj.age.stop, "step": obj.age.step},
            "race": obj.race,
        }
    raise TypeError(f"Type {type(obj)} not serializable")


def custom_deserializer(d: Any) -> Any:
    if "__type__" in d:
        if d["__type__"] == "ndarray":
            # Reconstruct numpy array from flattened data
            return np.array(d["data"]).reshape(d["shape"])
        if d["__type__"] == "float32":
            return np.float32(d["data"])
        if d["__type__"] == "range":  # Reconstruct range object
            return range(d["start"], d["stop"], d["step"])
        if d["__type__"] == "Face":
            return Face(
                bounding_box=np.array(d["bounding_box"]),
                score_set=d["score_set"],
                landmark_set={k: np.array(v) for k, v in d["landmark_set"].items()},
                angle=d["angle"],
                embedding=np.array(d["embedding"]),
                normed_embedding=np.array(d["normed_embedding"]),
                gender=d["gender"],
                age=range(d["age"]["start"], d["age"]["stop"], d["age"]["step"]),
                race=d["race"],
            )
    return d


# Updated read_json and write_json methods
def read_json(json_path: str) -> Optional[dict]:
    if is_file(json_path):
        try:
            with open(json_path, 'r') as json_file:
                return json.load(json_file, object_hook=custom_deserializer)
        except json.JSONDecodeError:
            pass
    return None


def write_json(json_path: str, content: dict) -> bool:
    with open(json_path, 'w') as json_file:
        json.dump(content, json_file, indent=4, default=custom_serializer)
    return is_file(json_path)
