"""
Dynamic dependency resolver for MIDAP.

All static project metadata (name, version, scripts, …) lives in pyproject.toml.
This file exists solely to supply install_requires at build time, because the Euler
cluster install sets MIDAP_INSTALL_VERSION=euler to select a different, fully-pinned
dependency set required for cluster compatibility.

Do NOT add static metadata here — edit pyproject.toml instead.
"""

import os
import platform

from setuptools import setup


def get_requirements():
    # ------------------------------------------------------------------ #
    # Euler cluster install                                                #
    # Triggered by:  export MIDAP_INSTALL_VERSION=euler && pip install -e #
    # All pins are intentional and must not be changed without testing on  #
    # the cluster. See euler/requirements.txt and euler/create_venv.sh.   #
    # ------------------------------------------------------------------ #
    if os.getenv("MIDAP_INSTALL_VERSION", "core").lower() == "euler":
        return [
            "btrack==0.4.6",
            "coverage>=7.3.2",
            "gitpython>=3.1.40",
            "napari[all]",
            "omnipose==0.4.4",
            "opencv-python>=4.8.1",
            "pandas>=2.0.2",
            "scikit-image>=0.19.3,<=0.20.0",
            "stardist>=0.8.5",
            "tensorflow==2.15.0",
            "tqdm>=4.65.0",
            "build",
            "twine",
            "jaraco.functools<=4.1.0",
        ]

    # ------------------------------------------------------------------ #
    # Standard (core) install                                              #
    # == pins replaced with >= + major-version upper bound so that future  #
    # major releases don't silently break the environment.                #
    # ------------------------------------------------------------------ #
    requirements = [
        "btrack>=0.7.0,<1",
        "cellpose>=3.1.1,<4",
        "coverage>=7.13.0,<8",
        "gitpython>=3.1.46,<4",
        "jupyterlab>=4.5.0,<5",
        "matplotlib>=3.10.0,<4",
        "napari[all]",
        "omnipose>=0.4.4,<1",
        "opencv-python>=4.13.0,<5",
        "pandas>=2.3.0,<3",
        "psutil>=5.9.8,<6",
        "pytest>=8.4.0,<9",
        "scikit-image>=0.25.0,<1",
        "stardist>=0.9.0,<1",
        "tensorflow>=2.18.0,<2.19",
        "tf-keras>=2.18.0,<2.19",
        "tqdm>=4.67.0,<5",
        "build",
        "twine",
    ]

    # Apple-silicon GPU acceleration.
    # tensorflow-metal must be paired with the installed TF minor version;
    # the <2 guard here matches the tensorflow<2.19 cap above (metal 1.x).
    if platform.processor() == "arm":
        requirements.append("tensorflow-metal>=1.0.0,<2")

    return requirements


setup(install_requires=get_requirements())
