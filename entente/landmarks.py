from cached_property import cached_property

class Landmarker(object):
    def __init__(self, source_mesh, landmarks):
        from cgal_search import require_cgal
        require_cgal()
        self.source_mesh = source_mesh
        self.landmarks = landmarks

    @classmethod
    def load(cls, source_mesh_path, landmark_path):
        from lace.mesh import Mesh
        from lace.serialization import meshlab_pickedpoints
        return cls(
            source_mesh=Mesh(filename=source_mesh_path),
            landmarks=meshlab_pickedpoints.load(landmark_path))

    @cached_property
    def regressor(self):
        import numpy as np
        from blmath.numerics.matlab import sparse
        from .geometry import compute_barycentric_coordinates
        from .cgal_search import faces_nearest_to_points

        landmark_points = np.array(self.landmarks.values())
        num_landmarks = len(landmark_points)

        face_indices = faces_nearest_to_points(self.source_mesh, landmark_points)
        vertex_indices = self.source_mesh.f[face_indices]
        coords = compute_barycentric_coordinates(
            self.source_mesh.v[vertex_indices],
            landmark_points)

        row_indices = np.repeat(np.arange(3*num_landmarks, step=3), 3) \
            .reshape(-1, 1) + np.arange(3, dtype=np.uint64)
        column_indices = 3 * np.repeat(vertex_indices, 3) \
            .reshape(-1, 3) + np.arange(3, dtype=np.uint64)
        tiled_coords = np.tile(coords.reshape(-1, 1), 3)
        return sparse(
            row_indices.flatten(),
            column_indices.flatten(),
            tiled_coords.flatten(),
            3*num_landmarks,
            3*self.source_mesh.v.shape[0])

    def invoke_regressor(self, target):
        coords = self.regressor * target.v.reshape(-1)
        return dict(zip(self.landmarks.keys(), coords.reshape(-1, 3)))

    def transfer_landmarks_onto(self, target):
        from .equality import have_same_topology
        if not have_same_topology(self.source_mesh, target):
            # raise ValueError('Target mesh must have the same topology')
            target.f = self.source_mesh.f
            target.ft = self.source_mesh.ft
        return self.invoke_regressor(target)
