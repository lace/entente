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


def deserialize_landmarks(landmark_data):
    return {point["name"]: np.array(point["point"]) for point in landmark_data}


def load_landmarks(landmark_path):
    with open(landmark_path, "r") as f:
        if landmark_path.endswith(".pp"):
            return deserialize_landmarks(try_load_meshlab_pickedpoints().load(f))
        else:
            serialized = json.load(f)
            return deserialize_landmarks(serialized)


def serialize_landmarks(landmarks):
    return [
        {"name": name, "point": point.tolist()} for (name, point) in landmarks.items()
    ]


def dump_landmarks(landmarks, landmark_path):
    with open(landmark_path, "w") as f:
        if landmark_path.endswith(".pp"):
            try_load_meshlab_pickedpoints().dump(serialize_landmarks(landmarks), f)
        else:
            json.dump(serialize_landmarks(landmarks), f)


def assert_landmarks_are_equal(first, second):
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[key], second[key]) for key in first) is True
