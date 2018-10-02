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


@cli.command()
@click.argument("tag")
def docker_build(tag):
    execute(
        "docker",
        "build",
        "-t",
        "laceproject/entente-ci-py2.7:{}".format(tag),
        "-f",
        "docker/entente-ci-py2.7/Dockerfile",
        ".",
    )


@cli.command()
@click.argument("tag")
def docker_push(tag):
    """
    When pushing a new version, bump the minor version. It's okay to re-push,
    though once it's being used in master, you should leave it alone.
    """
    execute("docker", "push", "laceproject/entente-ci-py2.7:{}".format(tag))


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
