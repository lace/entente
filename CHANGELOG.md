Changelog
=========

## 0.4.0 (Oct 4, 2018)

- Rework and improve entente.restore_correspondence.find_permutation and
  entente.restore_correspondence.restore_correspondence
    - Allow unmatched entries and rename find_permutation() to find_correspondence()
    - Flip semantics of a and b in find_correspondence()
    - Allow inexact matching by default
    - Improve documentation and clarity of tests
- Add entente.composite.composite_meshes

## 0.3.0 (Oct 2, 2018)

- Add entente.shuffle.shuffle_faces
- Add entente.shuffle.shuffle_vertices
- Add entente.restore_correspondence.find_permutation
- Add entente.restore_correspondence.restore_correspondence

## 0.2.0 (Oct 1, 2018)

- entente.cgal_search.faces_nearest_to_points: Rename argument `to_points` to
  `search_points`.
- entente.validation.validate_shape_from_ns: Wildcard arguments are now
  specified using `-1` instead of `'*'`.
- entente.landmarks.Landmarker: Make private `invoke_regressor` and
  `regressor`.
- Add docs on readthedocs.

## 0.1.0 (Sep 29, 2018)

Initial release.
