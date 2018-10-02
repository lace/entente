#!/usr/bin/env python

import click
import glob
from executor import execute


@click.group()
def cli():
    pass


@cli.command()
def init():
    execute("pip install --upgrade -r requirements_dev.txt")


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
    execute("nose2")


source_files = glob.glob("*.py") + ["entente/", "doc/"]


@cli.command()
def lint():
    execute("pyflakes", *source_files)


@cli.command()
def black():
    execute("black", *source_files)


@cli.command()
def black_check():
    execute("black", "--check", *source_files)


@cli.command()
def doc():
    execute("rm -rf build/ doc/build/ doc/api/")
    execute("sphinx-build -b html doc doc/build")


@cli.command()
def doc_open():
    execute("open doc/build/index.html")


@cli.command()
def upload():
    execute("rm -rf dist/")
    execute("python setup.py sdist")
    execute("twine upload dist/*")


if __name__ == "__main__":
    cli()
