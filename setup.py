import pathlib
from setuptools import setup, find_packages
from degann.networks.config_format import VERSION

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
with open(HERE / "README.md", "r", encoding="utf-8") as fh:
    README = fh.read()

DESCRIPTION = "Library for generating artificial neural networks for modeling the behavior of dynamic systems"
LONG_DESCRIPTION = README

# Setting up
setup(
    name="degann",
    version=VERSION,
    author="Pavel Alimov",
    author_email="<pashaalimov@gmail.com>",
    url="https://github.com/Krekep/degann",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "keras~=2.12.0",
        "matplotlib~=3.7.1",
        "numpy~=1.23.5",
        "scipy~=1.10.1",
        "tensorflow~=2.12.0",
    ],
    python_requires=">=3.10",
    keywords=["python", "ode", "differential equation", "neural network"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
