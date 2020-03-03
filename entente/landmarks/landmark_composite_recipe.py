from cached_property import cached_property
import lacecore
import numpy as np
from polliwog import Plane
import vg
from .landmarker import Landmarker
from .landmark_compositor import LandmarkCompositor
from ._mesh import DEFAULT_RADIUS


class LandmarkCompositeRecipe(object):
    def __init__(self, recipe):
        """
        Example recipe:

        base_mesh: examples/average.obj
        decimals: 2
        landmarks:
          - knee_left
          - knee_right
        symmetrize:
          reference_point: [0.5, 0.0, 0.0]
          normal: [1.0, 0.0, 0.0]
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
        self.decimals = recipe["decimals"]
        self.landmark_names = recipe["landmarks"]
        self.examples = recipe["examples"]
        self.symmetrize = recipe.get("symmetrize", None)

    @classmethod
    def load(cls, recipe_path):
        import yaml

        with open(recipe_path, "r") as f:
            recipe_data = yaml.safe_load(f)
        return cls(recipe_data)

    @cached_property
    def base_mesh(self):
        return lacecore.load_obj(self.base_mesh_path, triangulate=True)

    @property
    def _unsided_landmark_names(self):
        return list(
            set(
                [
                    x.replace("_right", "").replace("_left", "")
                    for x in self.landmark_names
                ]
            )
        )

    @cached_property
    def _plane_of_symmetry(self):
        return Plane(
            point_on_plane=np.array(self.symmetrize["reference_point"]),
            unit_normal=vg.normalize(np.array(self.symmetrize["normal"])),
        )

    def _symmetrize_landmarks(self, landmarks):
        from .symmetrize_landmarks import symmetrize_landmarks_using_plane

        result = {}
        for unsided_name in self._unsided_landmark_names:
            sided_names = [
                "{}_{}".format(unsided_name, side) for side in ["left", "right"]
            ]
            symmetrized = symmetrize_landmarks_using_plane(
                self._plane_of_symmetry, np.array([landmarks[k] for k in sided_names]),
            )
            for k, v in zip(sided_names, symmetrized):
                result[k] = v
        return result

    @cached_property
    def composite_landmarks(self):
        base_mesh = self.base_mesh
        compositor = LandmarkCompositor(
            base_mesh=base_mesh, landmark_names=self.landmark_names
        )
        for example in self.examples:
            example_mesh = lacecore.load_obj(example["mesh"], triangulate=True)
            landmarks = {k: example[k] for k in self.landmark_names}
            compositor.add_example(mesh=example_mesh, landmarks=landmarks)
        return compositor.result

    @cached_property
    def symmetrized_landmarks(self):
        return {
            k: v.tolist()
            for k, v in self._symmetrize_landmarks(self.composite_landmarks).items()
        }

    @cached_property
    def reprojected_landmarks(self):
        if self.symmetrize is None:
            landmarks = self.composite_landmarks
        else:
            landmarks = self.symmetrized_landmarks

        inverse_landmarker = Landmarker(source_mesh=self.base_mesh, landmarks=landmarks)

        reprojected = {}
        for example in self.examples:
            mesh = lacecore.load_obj(example["mesh"], triangulate=True)
            reprojected[example["id"]] = inverse_landmarker.transfer_landmarks_onto(
                mesh
            )
        return reprojected

    @property
    def original_and_reprojected_landmarks(self):
        result = {}
        for example in self.examples:
            example_id = example["id"]
            reprojected = self.reprojected_landmarks[example_id]
            result[example_id] = {
                k: {
                    "original": example[k],
                    "reprojected": reprojected[k].tolist(),
                    "displacement": np.round(
                        reprojected[k] - np.array(example[k]), decimals=self.decimals
                    ).tolist(),
                    "euclidean_distance": float(
                        round(
                            vg.euclidean_distance(np.array(example[k]), reprojected[k]),
                            ndigits=self.decimals,
                        )
                    ),
                }
                for i, k in enumerate(self.landmark_names)
            }
        return result

    def to_json(self):
        result = {
            "composited": self.composite_landmarks,
            "examples": self.original_and_reprojected_landmarks,
        }
        if self.symmetrize is not None:
            result["composited_and_symmetrized"] = self.symmetrized_landmarks
        return result

    def write_reprojected_landmarks(self, output_dir, radius=DEFAULT_RADIUS):
        import os
        from ._mesh import add_landmark_points

        original_and_reprojected_landmarks = self.original_and_reprojected_landmarks

        for example in self.examples:
            mesh = lacecore.load_obj(example["mesh"], triangulate=True)
            example_id = example["id"]

            these_original = [
                item["original"]
                for item in original_and_reprojected_landmarks[example_id].values()
            ]
            these_reprojected = [
                item["reprojected"]
                for item in original_and_reprojected_landmarks[example_id].values()
            ]
            add_landmark_points(
                mesh, np.vstack([these_original, these_reprojected]), radius=radius,
            )
            # Make the original landmarks blue, and the reprojected landmarks
            # the default red.
            e_color = [
                {
                    "color": (0, 0, 1),
                    "e_indices": np.arange(len(mesh.e) / 2, dtype=np.uint64),
                }
            ]

            out = os.path.join(output_dir, "{}.dae".format(example_id))
            mesh.write_dae(out, e_color)
