import importlib
from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    install_requires = f.read()

setup(
    name="entente",
    version=importlib.import_module("entente").__version__,
    description="Work with polygonal meshes which have vertex-wise correspondence",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Metabolize",
    author_email="github@paulmelnikow.com",
    url="https://github.com/metabolize/entente",
    project_urls={
        "Issue Tracker": "https://github.com/metabolize/entente/issues",
        "Documentation": "https://entente.readthedocs.io/en/stable/",
    },
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Topic :: Artistic Software",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
