from setuptools import setup, find_packages

version_info = {}
exec(open("entente/package_version.py").read(), version_info)


def load(filename):
    return open(filename, "rb").read().decode("utf-8")


setup(
    name="entente",
    version=version_info["__version__"],
    description="Work with polygonal meshes which have vertex-wise correspondence",
    long_description=load("README.md"),
    long_description_content_type="text/markdown",
    author="Metabolize",
    author_email="github@paulmelnikow.com",
    url="https://github.com/lace/entente",
    project_urls={
        "Issue Tracker": "https://github.com/lace/entente/issues",
        "Documentation": "https://entente.readthedocs.io/en/stable/",
    },
    packages=find_packages(),
    install_requires=load("requirements.txt"),
    extras_require={"landmarker": load("requirements_landmarker.txt")},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Artistic Software",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
