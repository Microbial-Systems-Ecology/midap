from setuptools import setup, find_packages
import platform
import os


def get_requirements(version = "core"):
    requirements = [
        "btrack==0.4.6",
        "coverage>=7.3.2",
        "gitpython>=3.1.40",
        "jupyterlab==4.0.8",
        "matplotlib>=3.3.0,<=3.6.3",
        "napari[all]",
        "omnipose==0.4.4",
        "opencv-python>=4.8.1",
        "pandas>=2.0.2",
        "psutil>=5.9.0",
        "pytest>=7.4.3",
        "scikit-image>=0.19.3,<=0.20.0",
        "stardist>=0.8.5",
        "tensorflow==2.13.0",
        "tqdm>=4.65.0",
        "build",
        "twine",
        "jaraco.functools<=4.1.0",
    ]

    if platform.processor() == "arm":
        requirements += ["tensorflow-metal"]
        
    # Check if installation is happening on euler
    if os.getenv("MIDAP_INSTALL_VERSION", "core").lower() == "euler":
        requirements = [
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

    return requirements


setup(
    name="midap",
    version="1.1.2",
    description="A package for cell segmentation and tracking.",
    long_description="""# MIDAP: Automated image segmentation and tracking for time-lapse microscopy of bacterial cells.

MIDAP is a flexible and user-friendly software for the automated analysis of
live-cell microscopy images of bacteria growing in a monolayer in microfluidics
chambers.

Through its graphical user interface, a selection of state-of-the-art
segmentation and tracking tools are provided, allowing the user to select the
most suited ones for their particular data set. Thanks to its modular
structure, additional segmentation and tracking tools can easily be integrated
as they are becoming available. After running the automated image analysis, the
user has the option to visually inspect and, if needed, manually correct
segmentation and tracking.

Documentation at https://github.com/Microbial-Systems-Ecology/midap/wiki
    """,
    long_description_content_type="text/markdown",
    author="Oschmann  Franziska, Fluri Janis",
    author_email="franziska.oschmann@id.ethz.ch, janis.fluri@id.ethz.ch",
    python_requires=">=3.9, <4",
    download_url="https://github.com/Microbial-Systems-Ecology/midap/releases/tag/0.3.18",
    keywords="Segmentation, Tracking, Biology",
    install_requires=get_requirements(),
    packages=find_packages(),
    include_package_data=True,
    project_urls={
        "Midap": "https://gitlab.ethz.ch/oschmanf/ackermann-bacteria-segmentation/"
    },
    entry_points={
        "console_scripts": [
            "midap = midap.main:run_module",
            "correct_segmentation = midap.apps.correct_segmentation:main",
            "midap_download = midap.apps.download_files:main",
        ],
    },
)
