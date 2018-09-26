import importlib
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    install_requires = f.read()

setup(
    name='entente',
    version=importlib.import_module('entente').__version__,
    description='Stretch polygonal meshes in segments along an axis',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Metabolize',
    author_email='github@paulmelnikow.com',
    url='https://github.com/metabolize/entente',
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Topic :: Artistic Software',
        'Topic :: Multimedia :: Graphics :: 3D Modeling',
        'Topic :: Scientific/Engineering :: Visualization'
    ]
)
