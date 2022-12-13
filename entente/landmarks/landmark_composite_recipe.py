from cached_property import cached_property
import lacecore
import numpy as np
from polliwog import Plane
from vg.compat import v1 as vg
from .landmark_compositor import LandmarkCompositor
from .landmarker import Landmarker
from .serialization import point_for_landmark_name

DEFAULT_RADIUS = 0.1


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
            reference_point=np.array(self.symmetrize["reference_point"]),
            normal=vg.normalize(np.array(self.symmetrize["normal"])),
        )

    def _symmetrize_landmarks(self, landmarks):
        from .symmetrize_landmarks import symmetrize_landmarks_using_plane

        for unsided_name in self._unsided_landmark_names:
            sided_names = [f"{unsided_name}_{side}" for side in ["left", "right"]]
            symmetrized = symmetrize_landmarks_using_plane(
                self._plane_of_symmetry,
                np.array(
                    [point_for_landmark_name(landmarks, name) for name in sided_names]
                ),
            )

        return [
            {
                "name": name,
                "point": new_point.tolist(),
            }
            for (name, new_point) in zip(sided_names, symmetrized)
        ]

    @cached_property
    def composite_landmarks(self):
        base_mesh = self.base_mesh
        compositor = LandmarkCompositor(
            base_mesh=base_mesh, landmark_names=self.landmark_names
        )
        for example in self.examples:
            example_mesh = lacecore.load_obj(example["mesh"], triangulate=True)
            landmarks = [
                {"name": name, "point": example[name]} for name in self.landmark_names
            ]
            compositor.add_example(mesh=example_mesh, landmarks=landmarks)
        return compositor.result

    @cached_property
    def symmetrized_landmarks(self):
        return self._symmetrize_landmarks(self.composite_landmarks)

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
            points = {}
            for name in self.landmark_names:
                reprojected_point = point_for_landmark_name(reprojected, name)
                points[name] = {
                    "original": example[name],
                    "reprojected": reprojected_point,
                    "displacement": np.round(
                        np.array(reprojected_point) - np.array(example[name]),
                        decimals=self.decimals,
                    ).tolist(),
                    "euclidean_distance": float(
                        round(
                            vg.euclidean_distance(
                                np.array(example[name]), np.array(reprojected_point)
                            ),
                            ndigits=self.decimals,
                        )
                    ),
                }
            result[example_id] = points
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
        from tri_again import Scene

        original_and_reprojected_landmarks = self.original_and_reprojected_landmarks

        for example in self.examples:
            example_id = example["id"]
            out = os.path.join(output_dir, f"{example_id}.dae")

            Scene(point_radius=radius).add_meshes(
                lacecore.load_obj(example["mesh"], triangulate=True)
            ).add_points(
                *[
                    item["original"]
                    for item in original_and_reprojected_landmarks[example_id].values()
                ],
                color="blue",
            ).add_points(
                *[
                    item["reprojected"]
                    for item in original_and_reprojected_landmarks[example_id].values()
                ],
                color="darkgreen",
            ).write(
                out
            )
