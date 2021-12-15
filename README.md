entente
=======

[![version](https://img.shields.io/pypi/v/entente?style=flat-square)][pypi]
[![python version](https://img.shields.io/pypi/pyversions/entente?style=flat-square)][pypi]
[![license](https://img.shields.io/pypi/l/entente?style=flat-square)][pypi]
[![coverage](https://img.shields.io/badge/coverage-100%25-brightgren?style=flat-square)][coverage]
[![build](https://img.shields.io/circleci/project/github/lace/entente/main?style=flat-square)][build]
[![docs build](https://img.shields.io/readthedocs/entente?style=flat-square)][docs build]
[![code style](https://img.shields.io/badge/code%20style-black-black?style=flat-square)][black]

Library for working with [lacecore][]-style polygonal meshes which are in
vertex-wise correspondence.

[pypi]: https://pypi.org/project/entente/
[coverage]: https://github.com/lace/entente/blob/main/.coveragerc
[black]: https://black.readthedocs.io/en/stable/
[lacecore]: https://github.com/metabolize/lacecore
[build]: https://circleci.com/gh/lace/entente/tree/main
[docs build]: https://entente.readthedocs.io/en/latest/


Features
--------

- Create a composite.
- Transfer landmarks from the surface of one mesh to the surface of another.
- Shuffle vertices.
- Restore correspondence of vertices.
- Spatial search, lightly wrapping [trimesh][].
- Complete documentation: https://entente.readthedocs.io/en/stable/



Installation
------------

To use the landmarker, run `pip install entente[surface_regressor]` which
installs [proximity][].

To use the CLI, run `pip install entente[surface_regressor,cli]` which
also installs [tri-again][] and pyyaml.


[proximity]: https://github.com/lace/proximity
[tri-again]: https://github.com/lace/tri-again


Usage
-----

```sh
python -m entente.cli transfer_landmarks source.obj source_landmarks.json target1.obj target2.obj ...
```

```yml
base_mesh: examples/average.obj
landmarks:
  - knee_left
  - knee_right
examples:
  - id: example01
    mesh: examples/example01.obj
    knee_left: [-10.0, 15.0, 4.0]
    knee_right: [10.0, 14.8, 4.1]
  - id: example02
    mesh: examples/example02.obj
    knee_left: [-11.0, 13.0, 3.5]
    knee_right: [12.0, 12.8, 3.4]
```

```sh
python -m entente.cli composite_landmarks recipe.yml
```


Development
-----------

First, [install Poetry][].

After cloning the repo, run `./bootstrap.zsh` to initialize a virtual
environment with the project's dependencies.

Subsequently, run `./dev.py install` to update the dependencies.

[install poetry]: https://python-poetry.org/docs/#installation


License
-------

The project is licensed under the MIT license.
