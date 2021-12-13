import numpy as np
import simplejson as json


def try_load_meshlab_pickedpoints():
    try:
        import meshlab_pickedpoints
    except ImportError:  # pragma: no cover
        raise ImportError(
            "To load Meshlab picked points files, install entente with the meshlab extra: "
            + "`pip install entente[surface_regressor,meshlab]`"
        )
    return meshlab_pickedpoints


def load_landmarks(landmark_path):
    with open(landmark_path, "r") as f:
        if landmark_path.endswith(".pp"):
            return try_load_meshlab_pickedpoints().load(f)
        else:
            serialized = json.load(f)
            return {item["name"]: np.array(item["point"]) for item in serialized}


def dump_landmarks(landmarks, landmark_path):
    with open(landmark_path, "w") as f:
        if landmark_path.endswith(".pp"):
            try_load_meshlab_pickedpoints().dump(landmarks, f)
        else:
            serialized = [
                {"name": name, "point": coords.tolist()}
                for name, coords in landmarks.items()
            ]
            json.dump(serialized, f)
