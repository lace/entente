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
        from ._trimesh_search import require_trimesh_with_rtree

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
        """
        Find the face each of the landmarks sits on, and then describe its
        position as a linear combination of the three vetices of that face.

        Represent this as a sparse matrix with a column for each coordinate
        of the source vertices and a row for each coordinate of the
        landmarks.

        Pushing target vertices through the matrix transfers the original
        landmarks to the target mesh.
        """
        import numpy as np
        from scipy.sparse import csc_matrix
        from polliwog.tri.barycentric import compute_barycentric_coordinates
        from ._trimesh_search import faces_nearest_to_points

        landmark_coords = np.array(list(self.landmarks.values()))
        face_indices = faces_nearest_to_points(self.source_mesh, landmark_coords)
        vertex_indices = self.source_mesh.f[face_indices]
        vertex_coeffs = compute_barycentric_coordinates(
            self.source_mesh.v[vertex_indices], landmark_coords
        )

        # Note the `.transpose()` at the end. The matrix is initially created
        # from data organized along columns of the result, not rows.
        values = np.repeat(vertex_coeffs, 3, axis=0).ravel()
        indices = (
            (3 * vertex_indices).reshape(-1, 1, 3)
            + np.arange(3, dtype=np.uint64).reshape(-1, 1)
        ).ravel()
        indptr = np.arange(len(values) + 1, step=3)
        shape = (3 * len(self.source_mesh.v), 3 * len(landmark_coords))
        return csc_matrix((values, indices, indptr), shape=shape).transpose()

    def transfer_landmarks_onto(self, target):
        """
        Transfer landmarks onto the given target mesh, which must be in the same
        topology as the source mesh.

        Args:
            target (lace.mesh.Mesh): Target mesh

        Returns:
            dict: A mapping of landmark names to a np.ndarray with shape `3x1`.
        """
        from ..equality import have_same_topology

        if not have_same_topology(self.source_mesh, target):
            raise ValueError("Target mesh must have the same topology")

        target_landmark_coords = (self._regressor * target.v.reshape(-1)).reshape(-1, 3)
        return dict(zip(self.landmarks.keys(), target_landmark_coords))
