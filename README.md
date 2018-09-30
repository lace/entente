entente
=======

[![version](https://img.shields.io/pypi/v/entente.svg?style=flat-square)][pypi]

Library for working with [lace][]-style polygonal meshes which have vertex-wise
correspondence.

[pypi]: https://pypi.org/project/entente/
[lace]: https://github.com/metabolize/lace


Features
--------

- Transfer landmarks from the surface of one mesh to the surface of another.


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
