entente
=======

[![version](https://img.shields.io/pypi/v/entente?style=flat-square)][pypi]
[![python version](https://img.shields.io/pypi/pyversions/entente?style=flat-square)][pypi]
[![license](https://img.shields.io/pypi/l/entente?style=flat-square)][pypi]
[![coverage](https://img.shields.io/badge/coverage-100%25-brightgren?style=flat-square)][coverage]
[![build](https://img.shields.io/circleci/project/github/lace/entente/master?style=flat-square)][build]
[![docs build](https://img.shields.io/readthedocs/entente?style=flat-square)][docs build]
[![code style](https://img.shields.io/badge/code%20style-black-black?style=flat-square)][black]

Library for working with [lace][]-style polygonal meshes which have vertex-wise
correspondence.

[pypi]: https://pypi.org/project/entente/
[coverage]: https://github.com/lace/entente/blob/master/.coveragerc
[black]: https://black.readthedocs.io/en/stable/
[lace]: https://github.com/metabolize/lace
[build]: https://circleci.com/gh/lace/entente/tree/master
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

```sh
pip install entente
```

[trimesh][], [rtree][], and [libspatialindex][] are required in order to use
landmark transfer and the AABB tree.

```sh
brew install spatialindex
pip install rtree trimesh
```

[pycollada][] is required for the landmark compositor.

```sh
pip install rtree trimesh
```

[trimesh]: https://trimsh.org/
[rtree]: http://toblerity.org/rtree/
[libspatialindex]: https://libspatialindex.org/
[pycollada]: https://github.com/pycollada/pycollada


Usage
-----

```sh
python -m entente.cli source.obj source.pp target1.obj target2.obj ...
```


Development
-----------

### Updating the Docker build for CircleCI

1. Make sure Docker is installed and running.
2. Build and push the images:

```sh
./dev.py docker-build 0.2.0  # Use the next available minor release.
./dev.py docker-push 0.2.0
```

3. Update the `image:` references in `.circleci/config.yml`.


Contribute
----------

- Issue Tracker: https://github.com/metabolize/entente/issues
- Source Code: https://github.com/metabolize/entente

Pull requests welcome!


Support
-------

If you are having issues, please let me know.


License
-------

The project is licensed under the MIT license.
