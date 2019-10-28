from setuptools import setup, find_packages

# Set version_info[__version__], while avoiding importing numpy, in case numpy
# and vg are being installed concurrently.
# https://packaging.python.org/guides/single-sourcing-package-version/
version_info = {}
exec(open("entente/package_version.py").read(), version_info)

readme = open("README.md", "rb").read().decode("utf-8")
install_requires = open("requirements.txt", "rb").read().decode("utf-8")

setup(
    name="entente",
    version=version_info["__version__"],
    description="Work with polygonal meshes which have vertex-wise correspondence",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Metabolize",
    author_email="github@paulmelnikow.com",
    url="https://github.com/lace/entente",
    project_urls={
        "Issue Tracker": "https://github.com/lace/entente/issues",
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
