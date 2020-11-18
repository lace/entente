# Changelog

## 1.0.0b1 (Nov. 18, 2020)

### New features

- Landmarker: Improve error message when passing in quad source or target
  meshes.

### Other changes

- Pin NumPy to avoid a regression in NumPy 1.19.
- Update meshlab-pickedpoints, lacecore, and polliwog dependencies.


## 1.0.0b0 (May 13, 2020)

### BREAKING CHANGES

- Replace lace with lacecore.
- Switch landmarker dependency from trimesh to proximity. libspatialindex is
  still required but rtree will be installed for you when you run
  `pip install entente[landmarker]`.
- Switch collada dependency to tri-again. pycollada is installed for you.
- Require polliwog 1.0.0b9.
- `shuffle_vertices()` and `shuffle_faces()` return new meshes. Optionally
  they also return the new ordering of the old elements, which is
  _the inverse of the index arrays these functions used to return_.

### New features

- Add `find_opposite_vertices()`.
- Add `symmetrize_landmarks()`.


## 0.11.0 (Dec 5, 2019)

### BREAKING CHANGES

- Require polliwog 1.0.0-beta.1.

### New features

- Add `find_rigid_transform()` and `find_rigid_rotation()` from [polliwog][].
- Landmark composite: Compare reprojected landmarks to the original, and
  output stats.

### Other improvements

- Use `yaml.safe_load()` and avoid PyYAML deprecation warning.


## 0.10.0 (Nov 25, 2019)

### BREAKING CHANGES

- Require Python 3.
- Require polliwog 0.12.0.


## 0.9.0 (Nov 1, 2019)

### BREAKING CHANGES

- Move `entente.landmarks.Landmarker` to
  `entente.landmarks.landmarker.Landmarker`.
- Make `trimesh_search` private.

### New features

- Add tool for compositing landmarks.

### Bug fixes

- Remove an undeclared dependency on `blmath`.
- Declare some missing dependencies.

### Other improvements

- 100% test coverage.
- Avoid generating documentation for test modules.
- Ensure test files are not shipped.

## 0.8.0 (Oct 27, 2019)

- Update dependencies.

## 0.7.1 (Sep 28, 2019)

- Unfork dependencies.

## 0.7.0 (Aug 30, 2019)

- Update for Python 3.

## 0.6.0 (Apr 3, 2019)

- Remove `entente.geometry` which has been moved to `polliwog.tri.barycentric`.
- Remove `entente.validation` which has been moved to `vg.shape.check`.

## 0.5.0 (Apr 3, 2019)

- Landmark using spatialindex, rtree, and trimesh, instead of CGAL.

## 0.4.1 (Nov 3, 2018)

- Remove unused dependency on pyyaml.

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


[polliwog]: https://github.com/lace/polliwog/
