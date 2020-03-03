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

    - The xyz coordinates of the comosite landmark on the base mesh
    """

    def __init__(self, base_mesh, landmark_names):
        self.base_mesh = base_mesh
        self.landmark_names = set(landmark_names)
        self.examples = []

    def add_example(self, mesh, landmarks):
        # By processing one mesh at a time, they don't all need to be loaded
        # into memory.
        if not set(landmarks.keys()).issuperset(self.landmark_names):
            raise ValueError(
                "Expected examples to contain keys {}".format(
                    ", ".join(self.landmark_names)
                )
            )
        landmarker = Landmarker(source_mesh=mesh, landmarks=landmarks)
        transferred = landmarker.transfer_landmarks_onto(self.base_mesh)
        self.examples.append(transferred)

    @property
    def result(self):
        return {
            name: np.average([example[name] for example in self.examples], axis=0)
            for name in self.landmark_names
        }
