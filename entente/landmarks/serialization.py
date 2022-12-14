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
            return serialized


def dump_landmarks(landmarks, landmark_path):
    with open(landmark_path, "w") as f:
        if landmark_path.endswith(".pp"):
            try_load_meshlab_pickedpoints().dump(landmarks, f)
        else:
            json.dump(landmarks, f)


def point_for_landmark_name(landmarks, name):
    return next(item for item in landmarks if item["name"] == name)["point"]
