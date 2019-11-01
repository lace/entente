from cached_property import cached_property
import numpy as np
from .landmarker import Landmarker
from .landmark_compositor import LandmarkCompositor


class LandmarkCompositeReceipe(object):
    def __init__(self, recipe):
        """
        Example recipe:

        base_mesh: examples/average.obj
        landmarks:
          - knee_left
          - knee_right
        examples:
          - id: example01
            mesh: examples/example01.obj
            knee_left: [-10.0, 15.0, 4.0]
            knee_right: [10.0, 14.8, 4.1]
          - id: example02
            mesh: examples/example02.obj
            knee_left: [-11.0, 13.0, 3.5]
            knee_right: [12.0, 12.8, 3.4]
        """
        self.base_mesh_path = recipe["base_mesh"]
        self.landmark_names = recipe["landmarks"]
        self.examples = recipe["examples"]

    @classmethod
    def load(cls, recipe_path):
        import yaml

        with open(recipe_path, "r") as f:
            recipe_data = yaml.load(f)
        return cls(recipe_data)

    @property
    def base_mesh(self):
        from lace.mesh import Mesh

        return Mesh(filename=self.base_mesh_path)

    @cached_property
    def composite_landmarks(self):
        from lace.mesh import Mesh

        compositor = LandmarkCompositor(
            base_mesh=self.base_mesh, landmark_names=self.landmark_names
        )
        for example in self.examples:
            example_mesh = Mesh(filename=example["mesh"])
            # `id` and `mesh` attrs are ignored.
            landmarks = example
            compositor.add_example(mesh=example_mesh, landmarks=landmarks)
        return compositor.result

    @cached_property
    def reprojected_landmarks(self):
        from lace.mesh import Mesh

        inverse_landmarker = Landmarker(
            source_mesh=self.base_mesh, landmarks=self.composite_landmarks
        )

        reprojected = {}
        for example in self.examples:
            mesh = Mesh(filename=example["mesh"])
            result[example["id"]] = inverse_landmarker.transfer_landmarks_onto()
        return reprojected

    def write_reprojected_landmarks(self, output_dir, radius=0.1):
        import os
        from lace.mesh import Mesh
        from ._mesh import add_landmark_points

        for example in self.examples:
            mesh = Mesh(filename=example["mesh"])
            example_id = example["id"]
            landmarks = np.array(self.reprojected_landmarks[example_id].values())
            add_landmark_points(mesh, landmarks, radius=radius)
            out = os.path.join(output_dir, "{}.dae".format(example_id))
            mesh.write_dae(out)
