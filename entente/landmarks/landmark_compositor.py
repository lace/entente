import numpy as np
from .landmarker import Landmarker


class LandmarkCompositor(object):
    """
    A tool for compositing landmarks from several examples in relation to
    a base mesh. Each example is projected onto the base mesh, then the
    points are averaged.

    The tool takes as input:

    - A base mesh
    - Several examples
        - Mesh (in correspondence with the base mesh)
        - xyz coordinates for one or more landmarks

    And will output:

    - The xyz coordinates of the composite landmark on the base mesh
    """

    def __init__(self, base_mesh, landmark_names):
        self.base_mesh = base_mesh
        self.landmark_names = set(landmark_names)
        self.examples = {name: [] for name in landmark_names}

    def add_example(self, mesh, landmarks):
        # By processing one mesh at a time, they don't all need to be loaded
        # into memory.
        landmark_names = [landmark["name"] for landmark in landmarks]
        if not set(landmark_names).issuperset(self.landmark_names):
            raise ValueError(
                f"Expected examples to contain keys {', '.join(self.landmark_names)}"
            )
        landmarker = Landmarker(source_mesh=mesh, landmarks=landmarks)
        transferred = landmarker.transfer_landmarks_onto(self.base_mesh)
        for point in transferred:
            self.examples[point["name"]].append(point["point"])

    @property
    def result(self):
        return [
            {
                "name": name,
                "point": np.average(np.array(self.examples[name]), axis=0).tolist(),
            }
            for name in self.landmark_names
        ]
