# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Setup configuration for MeshPrep v5."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "MeshPrep v5 - Automated mesh repair system"

setup(
    name="meshprep",
    version="5.0.0",
    description="Automated mesh repair system with ML and learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Allard Peper (Dragon Ace)",
    url="https://github.com/DragonAceNL/MeshPrep",
    license="Apache License 2.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24",
        "trimesh>=4.0",
        "click>=8.0",
        "scipy>=1.9",  # Required by trimesh
        "fast-simplification>=0.1",  # Required for mesh decimation
    ],
    extras_require={
        "ml": [
            "torch>=2.0",
            "torchvision>=0.15",
        ],
        "pymeshfix": ["pymeshfix>=0.16"],
        "open3d": ["open3d>=0.17"],
        "all": [
            "torch>=2.0",
            "torchvision>=0.15",
            "pymeshfix>=0.16",
            "open3d>=0.17",
        ],
    },
    entry_points={
        "console_scripts": [
            "meshprep=meshprep.cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering",
    ],
    keywords="3d mesh repair stl 3d-printing machine-learning",
)
