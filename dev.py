#!/usr/bin/env python

import click
from executor import execute

@click.group()
def cli():
    pass

@cli.command()
def init():
    execute('pip install -r requirements_dev.txt')

@cli.command()
def test():
    execute('nose2')

source_files = ['*.py', 'entente/']

@cli.command()
def black():
    execute('black', *source_files)

@cli.command()
def black_check():
    execute('black', '--check', *source_files)

@cli.command()
def upload():
    execute('rm -rf dist/')
    execute('python setup.py sdist')
    execute('twine upload dist/*')

if __name__ == '__main__':
    cli()
