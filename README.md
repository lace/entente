entente
=======

[![version](https://img.shields.io/pypi/v/entente.svg?style=flat-square)][pypi]
[![license](https://img.shields.io/pypi/l/entente.svg?style=flat-square)][pypi]
[![build](https://img.shields.io/circleci/project/github/lace/entente/master.svg?style=flat-square)][build]
[![docs build](https://img.shields.io/readthedocs/entente.svg?style=flat-square)][docs build]
[![code style](https://img.shields.io/badge/code%20style-black-black.svg?style=flat-square)][black]

Library for working with [lace][]-style polygonal meshes which have vertex-wise
correspondence.

[pypi]: https://pypi.org/project/entente/
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
- Spatial search, lightly wrapping CGAL.
- Complete documentation: https://entente.readthedocs.io/en/stable/


Installation
------------

Requires Python 2.7.

```sh
pip install entente
```

CGAL and cgal-bindings are required for landmark transfer and AABB tree. They
are optional because cgal-bindings takes a long time to build.

```sh
brew install cgal swig
pip install cgal-bindings
```


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

3. Update the `image:` references in .circleci/config.yml`.


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
