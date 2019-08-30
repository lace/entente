"""
Functions for transferring landmarks from one mesh to another.

This module requires libspatialindex and rtree. See note in `trimesh_search.py`.
"""

from cached_property import cached_property


class Landmarker(object):
    """
    An object which encapsulates a source mesh and a set of landmarks on that
    mesh. Its function is to transfer those landmarks onto a new mesh.

    The resultant landmarks will always be on or near the surface of the mesh.

    Args:
        source_mesh (lace.mesh.Mesh): The source mesh
        landmarks (dict): A mapping of landmark names to the points, which are
            `3x1` arraylike objects.
    """

    def __init__(self, source_mesh, landmarks):
        from .trimesh_search import require_trimesh_with_rtree

        require_trimesh_with_rtree()
        self.source_mesh = source_mesh
        self.landmarks = landmarks

    @classmethod
    def load(cls, source_mesh_path, landmark_path):
        """
        Create a landmarker using the given paths to a source mesh and landmarks.

        Args:
            source_mesh_path (str): File path to the source mesh.
            landmark_path (str): File path to a meshlab ``.pp`` file containing
                the landmark points.
        """
        from lace.mesh import Mesh
        from lace.serialization import meshlab_pickedpoints

        return cls(
            source_mesh=Mesh(filename=source_mesh_path),
            landmarks=meshlab_pickedpoints.load(landmark_path),
        )

    @cached_property
    def _regressor(self):
        import numpy as np
        from blmath.numerics.matlab import sparse
        from polliwog.tri.barycentric import compute_barycentric_coordinates
        from .trimesh_search import faces_nearest_to_points

        landmark_points = np.array(list(self.landmarks.values()))
        num_landmarks = len(landmark_points)

        face_indices = faces_nearest_to_points(self.source_mesh, landmark_points)
        vertex_indices = self.source_mesh.f[face_indices]
        coords = compute_barycentric_coordinates(
            self.source_mesh.v[vertex_indices], landmark_points
        )

        row_indices = np.repeat(np.arange(3 * num_landmarks, step=3), 3).reshape(
            -1, 1
        ) + np.arange(3, dtype=np.uint64)
        column_indices = 3 * np.repeat(vertex_indices, 3).reshape(-1, 3) + np.arange(
            3, dtype=np.uint64
        )
        tiled_coords = np.tile(coords.reshape(-1, 1), 3)
        return sparse(
            row_indices.flatten(),
            column_indices.flatten(),
            tiled_coords.flatten(),
            3 * num_landmarks,
            3 * self.source_mesh.v.shape[0],
        )

    def _invoke_regressor(self, target):
        coords = self._regressor * target.v.reshape(-1)
        return dict(zip(self.landmarks.keys(), coords.reshape(-1, 3)))

    def transfer_landmarks_onto(self, target):
        """
        Transfer landmarks onto the given target mesh, which must be in the same
        topology as the source mesh.

        Args:
            target (lace.mesh.Mesh): Target mesh

        Returns:
            dict: A mapping of landmark names to a np.ndarray with shape `3x1`.
        """
        from .equality import have_same_topology

        if not have_same_topology(self.source_mesh, target):
            raise ValueError("Target mesh must have the same topology")
        return self._invoke_regressor(target)
