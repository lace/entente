def add_landmark_points(mesh, coords, radius=0.01):
    from polliwog import Polyline
    import numpy as np

    offset = radius * np.eye(3)
    segments = np.repeat(coords, 6, axis=0).reshape(-1, 3, 2, 3)
    segments[:, :, 0] = segments[:, :, 0] + offset
    segments[:, :, 1] = segments[:, :, 1] - offset
    segments
    polylines = [
        Polyline(v=segment_vs, is_closed=False)
        for segment_vs in segments.reshape(len(coords) * 3, 2, 3)
    ]
    mesh.add_lines(polylines)