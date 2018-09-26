from cached_property import cached_property

class Landmarker(object):
    def __init__(self, source_mesh, landmarks):
        self.source_mesh = source_mesh
        self.landmarks = landmarks
        # self.landmarks = {
        #     'back bottom': landmarks['back bottom']
        # }
        self.landmarks['back bottom'][2] = -75

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

        landmark_points = np.array(self.landmarks.values())
        num_landmarks = len(landmark_points)

        face_indices, closest = self.source_mesh.cgal_closest_faces_and_points(
            landmark_points)
        vertex_indices, coefficients = self.source_mesh.barycentric_coordinates_for_points(
            landmark_points,
            face_indices)

        row_indices = np.repeat(np.arange(3*num_landmarks, step=3), 3) \
            .reshape(-1, 1) + np.arange(3, dtype=np.uint64)
        column_indices = 3 * np.repeat(vertex_indices, 3) \
            .reshape(-1, 3) + np.arange(3, dtype=np.uint64)
        values = np.tile(coefficients.reshape(-1, 1), 3)
        return sparse(
            row_indices.flatten(),
            column_indices.flatten(),
            values.flatten(),
            3*num_landmarks,
            3*self.source_mesh.v.shape[0])

    def invoke_regressor(self, target):
        coords = self.regressor * target.v.reshape(-1)
        return dict(zip(self.landmarks.keys(), coords.reshape(-1, 3)))

    def transfer_landmarks_onto(self, target):
        if not self.source_mesh.has_same_topology(target):
            # raise ValueError('Target mesh must have the same topology')
            target.f = self.source_mesh.f
            target.ft = self.source_mesh.ft

        return self.invoke_regressor(target)
