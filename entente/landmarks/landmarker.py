"""
Functions for transferring landmarks from one mesh to another.

This module requires entente to be installed with the `surface_regressor` extra:

    pip install entente[surface_regressor]
"""

from cached_property import cached_property
import numpy as np
from .serialization import load_landmarks


class Landmarker(object):
    """
    An object which encapsulates a source mesh and a set of landmarks on that
    mesh. Its function is to transfer those landmarks onto a new mesh.

    The resultant landmarks will always be on or near the surface of the mesh.

    Args:
        source_mesh (lacecore.Mesh): The source mesh
        landmarks (dict): A mapping of landmark names to the points, which are
            `3x1` arraylike objects.

    See also:
        `entente.path_transfer.PathTransfer`
    """

    def __init__(self, source_mesh, landmarks):
        if not source_mesh.is_tri:
            raise ValueError("Source mesh should be triangulated")

        self.source_mesh = source_mesh
        self.landmarks = landmarks

    @classmethod
    def load(cls, source_mesh_path, landmark_path):
        """
        Create a landmarker using the given paths to a source mesh and landmarks.

        Args:
            source_mesh_path (str): File path to the source mesh.
            landmark_path (str): File path to a JSON file or meshlab ``.pp``
                file containing the landmark points.
        """
        import lacecore

        return cls(
            source_mesh=lacecore.load_obj(source_mesh_path, triangulate=True),
            landmarks=load_landmarks(landmark_path),
        )

    @cached_property
    def _regressor(self):
        from ..surface_regressor import surface_regressor_for

        return surface_regressor_for(
            faces=self.source_mesh.f,
            source_mesh_vertices=self.source_mesh.v,
            query_points=np.array([point["point"] for point in self.landmarks]),
        )

    def transfer_landmarks_onto(self, target):
        """
        Transfer landmarks onto the given target mesh, which must be in the same
        topology as the source mesh.

        Args:
            target (lacecore.Mesh): Target mesh

        Returns:
            list: Points, using the Metabolize/Curvewise points JSON schema
        """
        from ..surface_regressor import apply_surface_regressor
        from ..equality import have_same_topology

        if not target.is_tri:
            raise ValueError("Target mesh must be triangulated")

        if not have_same_topology(self.source_mesh, target):
            raise ValueError("Target mesh must have the same topology")

        return [
            {
                "name": landmark["name"],
                "point": new_point.tolist(),
            }
            for (landmark, new_point) in zip(
                self.landmarks, apply_surface_regressor(self._regressor, target.v)
            )
        ]
