# Changelog (template)

## Legend for changelogs

- Feature : something that you couldn’t do before.
- Efficiency : an existing feature now may not require as much computation or memory.
- Fix : something that previously didn’t work as documentated – or according to reasonable expectations – should now work.

## Version number

Date

Feature: 
- ...

Efficiency:
- ...

Fix:
- ...

## [TBD]

TBD

Feature:
- Tracking tool
- Label images now uint16

Fix:
- Track ID of delta lineage started with 0, leading to a lost cell
- Fixed double assignment bug in delta tracking, where the same cell can be overwritten, i.e. be the successor for two cells
- Delta tracking added connectivity to lineage

Efficiency:
- Delta tracking now avoids resize of input
- Delta tracking data prep and cleanup much more efficient
- Delta tracking lower memory footprint

## [0.3.4]

2023-02-10

Feature:
- Bayes tracking: add tests for Bayes tracking
- Delta tracking: display expected runtime for tracking
- Delta tracking: reduce tracking results per processed time frime to reduce used memory

Fix:
- Bayes tracking: choose approximate update method if timeframe contains more than 1 000 cells
- Bayes tracking: get coordinates from label stack and not from tracking output (contained NaNs)
- Delta tracking: use binary segmentation for resize to avoid merge of cells
- Delta tracking: use original image size for tracking if cells are lost by image resize


## [0.3.3]

2023-02-09

Feature:
- File download now integrated into pipeline
- New CLI `midap_download` to manually download files


## [0.3.2]

2023-02-08

Feature:
- Env update for Windows support, now only support for python >= 3.9 


## [0.3.1]

2023-01-31

Feature:
- OME-Tif format now accepted


## [0.3.0]

2023-01-23

Feature:
- Automated testing for github
- Additional tests for the full pipeline
- Split installation for GPU and non-GPU support
- Mode READE contents to wiki
- Improved checkpointing for errors that kill the python process

Fix:
- Minor fixes discovered by tests


## [0.2.1]

2023-01-10

Feature:
- Removed all bash script, the pipeline can now be run via CLI `midap`
- The segmentation correction can also be invoked via CLI `segmentation_correction`
- New GUI layout for more flexibility
- New settings file type (config file)
- Advanced checkpointing options
- The pipeline always runs in batch mode
- Split tracking classes into two for easy DeltaType Subclassing
- Cleanup of Delta lineage
- New helper routines in utils
- All apps are now in the apps folder of the package and can be run with command line arguments as well.
- Cleanup of segmentation
- GUI based segmentation selection (#8) 
- Live cutout visualization (#5)

Fix:
- The test work again
- Bug fix issue #19
- Fix bug where Lineage ID was always the same as the tracking ID 


## [0.2.0]

2022-12-15

Feature:
- Added new tracking algorithm: Bayesian tracking

Fix:
- Adjusted segmentation postprocessing to use same steps during model selection and segmentation of image stacks

## [0.1.7]

2022-12-01

Feature:
- Added scripts to run jupyter on Euler with SLURM and GPU support.
- Updated README in base directory and created README for Euler.

## [0.1.6]

2022-11-30

Feature:
- Introduced testing to most of the important pipeline routines.
- Consistent type hints and docstring of the tracking files.

Fix:
- Minor bug fixes according to tests.

## [0.1.5]

2022-11-17

Feature:
- moved all networks into a designated `networks` folder in the package.
- Added support for the old delta model for tracking
- Refactored to standard UNet into a class
- All base classes are now abstract base classes that enforce setting the required methods
- Added the `training` directory for the training and finetuning of models.
- Added example notebooks and a README about the training of custom models.
- Added basic UNet block into the `netoworks` directory that can be used for custom models.

Fix:
- Cleanup of the base classes, removal of redundant methods
- Minor fix in the preprocessing where some pixels of the input image weren't used
- Fix in the weight_map generation that could case the cell border to receive zero weight during the training
- Fix in data augmentation where the same augmentation was applied multiple times.


## [0.1.4]

2022-11-11

Feature:
- Added tool to investigate quality of segmentations (all frames of one tif-stack) and open napari for manual correction if needed

Fix:
- Generated requirements.txt including napari

## [0.1.3]

2022-11-02

Feature:
- Added Tracking modularity in the same way as Segmentation and image cutout modularity via subclasses
- The pipeline accepts a `--cpu_only` flag that will set CUDA_VISIBLE_DEVICES to -1

Fix:
- Minor typo in README
- String comparison performed to check if the --restart argument has been supplied with a path was faulty


## [0.1.2]

2022-11-01

Feature:
- Omnipose support for non M1 local machines.

Fix:
- Updated the creation scripts of the euler env to use python 3.8.5, similar to the conda env
- Minor change in `run_pipeline.sh` such that the pipeline runs with tracking only

## [0.1.1]

2022-10-27

Feature:
- Integrate package install into the env creation, update README accordingly
- Notebook with visualization examples of tracking results

Fix:
- Update requirements for the M1 Mac


## [0.1.0]

2022-10-17

Feature:
- Segmentation postprocessing removes noise in the segmentation based on the average size of the segmented object instead
of the percentile. It is now possible to keep all objects.

Fix:
- Tracking and lineage can now deal with emtpy files/results
- Moved frame restriction in bash file such that is runs after the CHECKDIR is sourced in the setup

Efficiency:
- RAM usage integrated in progressbar in tracking
- Some cleanup


## [0.0.5]

2022-10-13

Feature:
- Integrated M1 workflow into the main script such that one can run everything from one script
- Added support for Euler setup
- Add instruction for both to README, for now Euler only with LSF
- Removed matlab suppersegger and modules

Fix:
- Open CV dependency was not in the requirements

## [0.0.4]

2022-10-10

Feature:
- Changes the `src` folder to `midap` and transformed the repo into packages
- Added modularity support for chamber cutting and cell segmentation
- Tracking output, segmentation output and raw images are stores as separate h5-files
- Cell lineages are stored in csv-file containing labelID, lineageID, trackID, IDs of mother and daughter cells, and features (area, length, intensity etc) per cell
- Update of cell tracking algorithm to DELTA2.0 and application of model weights generated during training on patches
- Improved postprocessing of tracking: cleaning of tracking results by finding segmented cell which has largest overlap

Fix:
- Minor fix in tracking to ensure that trackID_list is at least as long as global_IDs

## [0.0.3]

2022-10-05

Feature:
- Pipeline supports running PH + N Channels via comma separated list of channels in the GUI
- Changed backend of matplolib to be compatible with CentOS clusters
- Added logging utils for standardized logging
- Started moving scripts to apps folder

Fix:
- Added quotes to support file names with spaces
- Fix bug where local bash variables with interpolated commands (`local VAR=$(...)`) are not trapped when written on one line
- Set -maxdepth of find command in copy_files to 1 to avoid finding more than one file in the datasets
- Catch Error if no identifier was found
- Added an exeption if error was out of callstack

Efficiency:
- Cleanup of files in apps folder and their related imports in src
- Preparation of cut chambers and segmentation for user defined subclasses


## [0.0.2]

2022-09-16

Feature:
- `set_parameters.py` puts a header into `setting.sh` containing the current date, time and the git hash of the repo
- Additional metadata like the cutout corners or chosen models are saved into `settings.sh`
- `settings.sh` is copied to each data folder in the FAMILY\_MACHINE setting for reproducibility
- New argument `--loglevel` for the bash script to choose verbosity of output
- `--restart` flag now accepting an optional path to restart from other directories

Fix:
- Tensorflow output is suppressed by default, change `TF_CPP_MIN_LOG_LEVEL` in bash script to increase
- Variables that are already set in `settings.sh` are updated instead of appended at the end
- Moved last copy settings inside tracking function to avoid unintened overwrite for restarts
- Fix bug in cut images where an alignment offset of 0 caused a crash

## [0.0.1]

2022-09-13

Feature:
- bash script `run_pipeline_test.sh` accecpting inputs
- pipeline can be restarted from checkpoints
- bash script has logging

Fix:
- GUI elements `restrict_frames.py` and `restrict_frames.py` exit with code 1 when "Cancel" or "X" button is pressed.


