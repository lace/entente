class Landmarker(object):
    def __init__(self, source_mesh, landmarks):
        self.source_mesh = source_mesh
        self.landmarks = landmarks
        # self.landmarks = {
        #     'back bottom': landmarks['back bottom']
        # }
        # self.landmarks['back bottom'][2] = -75

    @classmethod
    def load(cls, source_mesh_path, landmark_path):
        from lace.mesh import Mesh
        from lace.serialization import meshlab_pickedpoints
        return cls(
            source_mesh=Mesh(filename=source_mesh_path),
            landmarks=meshlab_pickedpoints.load(landmark_path))

    def transfer_landmarks_onto(self, target):
        import numpy as np
        from blmath.numerics.matlab import col, sparse

        if not self.source_mesh.has_same_topology(target):
            target.f = self.source_mesh.f
            target.ft = self.source_mesh.ft
            # target.f = self.source_mesh.f
            # import pdb
            # pdb.set_trace()
            # raise ValueError('Target mesh must have the same topology')

        # return dict(zip(self.landmarks.keys(), closest_points))
        # return {
        #     k: target.v[target.f[face_indices[i]]]
        #     for i, k in enumerate(self.landmarks.keys())
        # }
        landmark_points = np.array(self.landmarks.values())
        face_indices, closest = self.source_mesh.cgal_closest_faces_and_points(landmark_points)
        print closest
        print face_indices
        vertex_indices, coefficients = self.source_mesh.barycentric_coordinates_for_points(
            landmark_points,
            face_indices,
            project_to_face=False)
        vertex_indices2, coefficients2 = self.source_mesh.barycentric_coordinates_for_points(
            landmark_points,
            face_indices)

        result = {}
        for i, k in enumerate(self.landmarks.keys()):
            result[k] = np.matmul(self.source_mesh.v[self.source_mesh.f[face_indices[i]]].T, coefficients[i])

        print result


        # import pdb
        # pdb.set_trace()

        column_indices = np.hstack([
            col(3*vertex_indices + i)
            for i in range(3)
        ]).reshape(-1)
        # column_indices = vertex_indices.T.ravel()
        row_indices = np.hstack([
            [3*index, 3*index + 1, 3*index + 2] * 3
            for index in range(len(landmark_points))
        ])
        # np.hstack([
        #     [0, 1, 2] * 3,
        #     [3, 4, 5] * 3,
        #     [6, 7, 8] * 3,
        # ])
        # row_indices = np.arange(3 * len(landmark_points))
        values = np.hstack([
            col(coefficients)
            for i in range(3)
        ]).flatten()
        # values = coefficients.T.ravel()
        # import pdb
        # pdb.set_trace()
        regressor = sparse(
            row_indices,
            column_indices,
            values,
            3*len(landmark_points),
            len(self.source_mesh.v.reshape(-1)))

        import pdb
        pdb.set_trace()

        res = (regressor * target.v.reshape(-1)).reshape(-1, 3)
        return dict(zip(self.landmarks.keys(), res))
