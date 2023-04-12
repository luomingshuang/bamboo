#!/usr/bin/env python3

from pathlib import Path

from setuptools import find_packages, setup

bamboo_dir = Path(__file__).parent
install_requires = (bamboo_dir / "requirements.txt").read_text().splitlines()

setup(
    name="bamboo",
    version="1.0",
    python_requires=">=3.6.0",
    description="Some Interesting Algorithms For Computer Vision.",
    author="Mingshuang Luo",
    license="Apache-2.0 License",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
)
