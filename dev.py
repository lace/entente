#!/usr/bin/env python

import os
import click
from executor import execute


def python_source_files():
    import glob

    return glob.glob("*.py") + ["entente/", "doc/"]


@click.group()
def cli():
    pass


@cli.command()
def init():
    execute("pip install --upgrade -r requirements_dev.txt")


@cli.command()
def clean():
    execute("find . -name '*.pyc' -or -name '__pycache__' -delete")


def docker_repo(python_version, tag):
    return "laceproject/entente-ci-{}:{}".format(python_version, tag)


python_versions = ["py3.6", "py2.7"]


@cli.command()
@click.argument("tag")
def docker_build(tag):
    for python_version in python_versions:
        execute(
            "docker",
            "build",
            "-t",
            docker_repo(python_version, tag),
            "-f",
            "docker/entente-ci/Dockerfile.{}".format(python_version),
            ".",
        )


@cli.command()
@click.argument("tag")
def docker_push(tag):
    """
    When pushing a new version, bump the minor version. It's okay to re-push,
    though once it's being used in master, you should leave it alone.
    """
    for python_version in python_versions:
        execute("docker", "push", docker_repo(python_version, tag))


@cli.command()
def test():
    execute("python3 -m pytest")


@cli.command()
def coverage():
    execute("python3 -m pytest --cov=entente")


@cli.command()
def coverage_py2():
    execute("python2 -m pytest --cov=entente")


@cli.command()
def coverage_report():
    execute("coverage html")
    execute("open htmlcov/index.html")


@cli.command()
def lint():
    execute("flake8", *python_source_files())


@cli.command()
def black():
    execute("black", *python_source_files())


@cli.command()
def black_check():
    execute("black", "--check", *python_source_files())


@cli.command()
def doc():
    execute("rm -rf build/ doc/build/ doc/api/")
    execute("sphinx-build -b html doc doc/build")


@cli.command()
def doc_open():
    execute("open doc/build/index.html")


@cli.command()
def publish():
    execute("rm -rf dist/")
    execute("python setup.py sdist")
    execute("twine upload dist/*")


if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    cli()
