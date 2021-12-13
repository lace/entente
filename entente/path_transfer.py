from cached_property import cached_property
from polliwog import Polyline


class PathTransfer:
    """
    Transfer a path drawn on the surface of a mesh to other meshes which are in
    correspondence.

    See also:
        `entente.landmarking.Landmarker`

    Args:
        source_mesh (lacecore.Mesh): The source mesh
        source_path (polliwog.Polyline): The source path
    """

    def __init__(self, source_mesh, source_path):
        from lacecore import Mesh

        assert isinstance(source_mesh, Mesh)
        self.source_mesh = source_mesh

        assert isinstance(source_path, Polyline)
        self.source_path = source_path

    @cached_property
    def _regressor(self):
        from .surface_regressor import surface_regressor_for

        return surface_regressor_for(
            faces=self.source_mesh.f,
            source_mesh_vertices=self.source_mesh.v,
            query_points=self.source_path.v,
        )

    def path_for(self, target_mesh):
        """
        Transfer landmarks onto the given target mesh, which must be in the same
        topology as the source mesh.

        Args:
            target_mesh (lacecore.Mesh): Target mesh

        Returns:
            polliwog.Polyline: The path transferred to the target mesh.
        """
        from .surface_regressor import apply_surface_regressor
        from .equality import have_same_topology

        if not target_mesh.is_tri:
            raise ValueError("Target mesh must be triangulated")

        if not have_same_topology(self.source_mesh, target_mesh):
            raise ValueError("Target mesh must have the same topology")

        return Polyline(
            v=apply_surface_regressor(self._regressor, target_mesh.v),
            is_closed=self.source_path.is_closed,
        )
