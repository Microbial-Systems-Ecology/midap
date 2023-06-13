import argparse
import os
import shutil
import sys
from glob import glob
from pathlib import Path
from shutil import copyfile

import numpy as np
import pkg_resources


def run_module(args=None):
    """
    Runs the module, i.e. the cell segmentation and tracking pipeline.
    :param args: arguments to parse, defaults to reading in the arguments from the standard input (command line)
    """

    # get the args
    if args is None:
        args = sys.argv[1:]

    # argument parsing
    description = "Runs the cell segmentation and tracking pipeline."
    parser = argparse.ArgumentParser(description=description, add_help=True,
                                     usage="midap [-h] [--restart [RESTART]] [--headless] "
                                           "[--loglevel LOGLEVEL] [--cpu_only] [--create_config]")
    # This arge is default if the flag is not set, it is const if it is set without arg, and it is the arg if provided
    parser.add_argument("--restart", nargs="?", default=None, const='.', type=str,
                        help="Restart pipeline from log file. If a path is specified the checkpoint and settings file "
                             "will be restored from the path, otherwise the current working directory is searched.")
    parser.add_argument("--headless", action="store_true",
                        help="Run pipeline in headless mode, ALL parameters have to be set prior in a config file.")
    parser.add_argument("--loglevel", type=int, default=7,
                        help="Set logging level of script (0-7), defaults to 7 (max log)")
    parser.add_argument("--cpu_only", action="store_true",
                        help="Sets CUDA_VISIBLE_DEVICES to -1 which will cause most! applications to use CPU only.")
    parser.add_argument("--create_config", action="store_true",
                        help="If this flag is set, all other arguments will be ignored and a 'settings.ini' config "
                             "file is generated in the current working directory. This option is meant generate "
                             "config file templates for the '--headless' mode. Note that this will overwrite "
                             "if a file already exists.")
    parser.add_argument("--prepare_config_cluster", action="store_true",
                    help="This option is meant to generate a config file and the output folder structure for a "
                            "dataset. The output folder is decompressed and can be uploaded to the cluster to "
                            "continue the pipeline in the '--headless' mode. Note that this will overwrite "
                            "if a file already exists.")

    # parsing
    args = parser.parse_args(args)

    # Some constants or conventions
    config_file = "settings.ini"
    check_file = "checkpoints.log"
    raw_im_folder = "raw_im"
    cut_im_folder = "cut_im"
    seg_im_folder = "seg_im"
    seg_im_bin_folder = "seg_im_bin"
    track_folder = "track_output"

    # Argument handling
    ###################

    # we do the local imports here to intercept TF import
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # supress TF blurp
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # set the global loglevel
    os.environ["__VERBOSE"] = str(args.loglevel)

    # create a logger
    from midap.utils import get_logger
    logger = get_logger("MIDAP", args.loglevel)

    # print the version
    version = pkg_resources.require("midap")[0].version
    logger.info(f"Running MIDAP version: {version}")

    # imports
    logger.info(f"Importing all dependencies...")
    from midap.checkpoint import Checkpoint, CheckpointManager
    from midap.config import Config
    from midap.apps import download_files, init_GUI, split_frames, cut_chamber, segment_cells, segment_analysis, track_cells
    logger.info("Done!")

    # Download the files if necessary
    logger.info(f"Checking necessary files...")
    download_files.main(args=[])
    logger.info("Done!")

    # create a config file if requested and exit
    if args.create_config:
        config = Config(fname="settings.ini")
        config.to_file(overwrite=True)
        return 0

    # check if we are restarting
    restart = False
    if args.restart is not None:
        # we set the restart flag
        restart = True

        # we search for the checkpoint file (recursive subdirectory search)
        checkpoints = glob(os.path.join(args.restart, "**", check_file), recursive=True)

        # exception handling
        if len(checkpoints) == 0:
            raise FileNotFoundError(f"No checkpoint found in the restart directory: {args.restart}")
        if len(checkpoints) > 1:
            raise ValueError(f"Multiple checkpoints found in the restart directory: {args.restart}")

        # we extract the checkpoing
        prev_checkpoint = Path(checkpoints[0]).absolute()
        logger.info(f"Found checkpoint: {prev_checkpoint}")

        # now we get the corresponding config file
        prev_config = prev_checkpoint.parent.joinpath(config_file)
        if not prev_config.exists():
            raise FileNotFoundError(f"No corresponding config file found: {prev_config}")
        logger.info(f"Found corresponding config file: {prev_config}")

        # we copy the files to the current working directory (if necessary)
        try:
            copyfile(prev_config, config_file)
            copyfile(prev_checkpoint, check_file)
        except shutil.SameFileError:
            pass

        # now we create the config and checkpoint (we do a full check if we are in headless mode)
        config = Config.from_file(config_file, full_check=args.headless)
        checkpoint = Checkpoint.from_file(check_file)

    # since we do a full check above if we have headless mode and restart this is an elif
    elif args.headless:
        # read the config from the working directory
        logger.info("Running in headless mode, checking config file...")
        config = Config.from_file(config_file, full_check=True)
        checkpoint = Checkpoint(check_file)

    # we are not restarting nor are we in headless mode
    else:
        logger.info("Starting up initial GUI...")
        # start the GUI
        init_GUI.main(config_file=config_file)
        # load in the config and create a checkpoint
        config = Config.from_file(fname=config_file, full_check=False)
        checkpoint = Checkpoint(check_file)

    # Setup
    #######

    # get the current base folder
    base_path = Path(config.get("General", "FolderPath"))

    # we cycle through all pos identifiers
    for identifier in config.getlist("General", "IdentifierFound"):
        # read out what we need to do
        run_segmentation = config.get(identifier, "RunOption").lower() in ['both', 'segmentation']
        # current path of the identifier
        current_path = base_path.joinpath(identifier)

        # stuff we do for the segmentation
        if run_segmentation:

            # setup all the directories
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="SetupDirs",
                                   identifier=identifier) as checker:
                # check to skip
                checker.check()

                logger.info(f"Generating folder structure for {identifier}")

                # remove the folder if it exists
                if current_path.exists():
                    shutil.rmtree(current_path, ignore_errors=False)

                # we create all the necessary directories
                current_path.mkdir(parents=True)

                # channel directories
                for channel in config.getlist(identifier, "Channels"):
                    current_path.joinpath(channel, raw_im_folder).mkdir(parents=True)
                    current_path.joinpath(channel, cut_im_folder).mkdir(parents=True)
                    current_path.joinpath(channel, seg_im_folder).mkdir(parents=True)
                    current_path.joinpath(channel, seg_im_bin_folder).mkdir(parents=True)
                    current_path.joinpath(channel, track_folder).mkdir(parents=True)

            # copy the files
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="CopyFiles",
                                   identifier=identifier, copy_path=current_path) as checker:
                # check to skip
                checker.check()

                logger.info(f"Copying files for {identifier}")

                # we get all the files in the base bath that match
                file_ext = config.get("General", "FileType")
                if file_ext == "ome.tif":
                    files = base_path.glob(f"*{identifier}*/**/*.ome.tif")
                else:
                    files = base_path.glob(f"*{identifier}*.{file_ext}")
                for fname in files:
                    for channel in config.getlist(identifier, "Channels"):
                        if channel in fname.stem:
                            logger.info(f"Copying '{fname.name}'...")
                            copyfile(fname, current_path.joinpath(channel, fname.name))

            # This is just to fill in the config file, i.e. split files 2 frames, get corners, etc
            ######################################################################################

            # split frames
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="SplitFramesInit",
                                   identifier=identifier, copy_path=current_path) as checker:
                # check to skip
                checker.check()

                logger.info(f"Splitting test frames for {identifier}")

                # split the frames for all channels
                file_ext = config.get("General", "FileType")
                for channel in config.getlist(identifier, "Channels"):
                    paths = list(current_path.joinpath(channel).glob(f"*.{file_ext}"))
                    if len(paths) == 0:
                        raise FileNotFoundError(f"No file of the type '.{file_ext}' exists for channel {channel}")
                    if len(paths) > 1:
                        raise FileExistsError(f"More than one file of the type '.{file_ext}' "
                                              f"exists for channel {channel}")

                    # we only get the first frame and the mid frame
                    first_frame = config.getint(identifier, "StartFrame")
                    mid_frame = int(0.5*(first_frame + config.getint(identifier, "EndFrame")))
                    frames = np.unique([first_frame, mid_frame])
                    split_frames.main(path=paths[0], save_dir=current_path.joinpath(channel, raw_im_folder),
                                      frames=frames,
                                      deconv=config.get(identifier, "Deconvolution"),
                                      loglevel=args.loglevel)

            # cut chamber and images
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="CutFramesInit",
                                   identifier=identifier, copy_path=current_path) as checker:
                # check to skip
                checker.check()

                logger.info(f"Cutting test frames for {identifier}")

                # get the paths
                paths = [current_path.joinpath(channel, raw_im_folder)
                         for channel in config.getlist(identifier, "Channels")]

                # Do the init cutouts
                if config.get(identifier, "Corners") == "None":
                    corners = None
                else:
                    corners = tuple([int(corner) for corner in config.getlist(identifier, "Corners")])
                cut_corners = cut_chamber.main(channel=paths, cutout_class=config.get(identifier, "CutImgClass"),
                                               corners=corners)

                # save the corners if necessary
                if corners is None:
                    corners = f"{cut_corners[0]},{cut_corners[1]},{cut_corners[2]},{cut_corners[3]}"
                    config.set(identifier, "Corners", corners)
                    config.to_file()

            # select the networks
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="SegmentationInit",
                                   identifier=identifier, copy_path=current_path) as checker:
                # check to skip
                checker.check()

                logger.info(f"Segmenting test frames for {identifier}...")

                # cycle through all channels
                for num, channel in enumerate(config.getlist(identifier, "Channels")):
                    # The phase channel is always the first
                    if num == 0 and not config.getboolean(identifier, "PhaseSegmentation"):
                        continue

                    # get the current model weight (if defined)
                    model_weights = config.get(identifier, f"ModelWeights_{channel}", fallback=None)

                    # run the selector
                    segmentation_class = config.get(identifier, "SegmentationClass")
                    if segmentation_class == "HybridSegmentation":
                        path_model_weights = Path(__file__).parent.parent.joinpath("model_weights",
                                                                                   "model_weights_hybrid")
                    else:
                        path_model_weights = Path(__file__).parent.parent.joinpath("model_weights",
                                                                                   "model_weights_legacy")
                    weights = segment_cells.main(path_model_weights=path_model_weights, path_pos=current_path,
                                                 path_channel=channel, postprocessing=True, network_name=model_weights,
                                                 segmentation_class=segmentation_class, just_select=True,
                                                 img_threshold=config.getfloat(identifier, "ImgThreshold")),

                    # save to config
                    if model_weights is None:
                        config.set(identifier, f"ModelWeights_{channel}", weights)
                        config.to_file()

    # we cycle through all pos identifiers again to perform all tasks fully
    #######################################################################

    for identifier in config.getlist("General", "IdentifierFound"):
        # read out what we need to do
        run_segmentation = config.get(identifier, "RunOption").lower() in ['both', 'segmentation']
        run_tracking = config.get(identifier, "RunOption").lower() in ['both', 'tracking']
        # current path of the identifier
        current_path = base_path.joinpath(identifier)

        # stuff we do for the segmentation
        if run_segmentation:
            # split frames
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="SplitFramesFull",
                                   identifier=identifier, copy_path=current_path) as checker:

                # exit if this is only run to prepare config
                if args.prepare_config_cluster:
                    sys.exit('Preparation of config file is finished. Please follow instructions on https://github.com/Microbial-Systems-Ecology/midap/wiki/MIDAP-On-Euler to submit your job on the cluster.')

                # check to skip
                checker.check()

                logger.info(f"Splitting all frames for {identifier}")

                # split the frames for all channels
                file_ext = config.get("General", "FileType")
                for channel in config.getlist(identifier, "Channels"):
                    paths = list(current_path.joinpath(channel).glob(f"*.{file_ext}"))
                    if len(paths) > 1:
                        raise FileExistsError(f"More than one file of the type '.{file_ext}' "
                                              f"exists for channel {channel}")

                    # get all the frames and split
                    frames = np.arange(config.getint(identifier, "StartFrame"), config.getint(identifier, "EndFrame"))
                    split_frames.main(path=paths[0], save_dir=current_path.joinpath(channel, raw_im_folder),
                                      frames=frames,
                                      deconv=config.get(identifier, "Deconvolution"),
                                      loglevel=args.loglevel)

            # cut chamber and images
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="CutFramesFull",
                                   identifier=identifier, copy_path=current_path) as checker:
                # check to skip
                checker.check()

                logger.info(f"Cutting all frames for {identifier}")

                # get the paths
                paths = [current_path.joinpath(channel, raw_im_folder)
                         for channel in config.getlist(identifier, "Channels")]

                # Get the corners and cut
                corners = tuple([int(corner) for corner in config.getlist(identifier, "Corners")])
                _ = cut_chamber.main(channel=paths, cutout_class=config.get(identifier, "CutImgClass"), corners=corners)

            # run full segmentation (we checkpoint after each channel)
            for num, channel in enumerate(config.getlist(identifier, "Channels")):
                # The phase channel is always the first
                if num == 0 and not config.getboolean(identifier, "PhaseSegmentation"):
                    continue

                with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config,
                                       state=f"SegmentationFull_{channel}", identifier=identifier,
                                       copy_path=current_path) as checker:
                    # check to skip
                    checker.check()

                    logger.info(f"Segmenting all frames for {identifier} and channel {channel}...")

                    # get the current model weight (if defined)
                    model_weights = config.get(identifier, f"ModelWeights_{channel}")

                    # run the segmentation, the actual path to the weights does not matter anymore since it is selected
                    path_model_weights = Path(__file__).parent.parent.joinpath("model_weights")
                    _ = segment_cells.main(path_model_weights=path_model_weights, path_pos=current_path,
                                           path_channel=channel, postprocessing=True, network_name=model_weights,
                                           segmentation_class=config.get(identifier, "SegmentationClass"),
                                           img_threshold=config.getfloat(identifier, "ImgThreshold"))
                    # analyse the images
                    segment_analysis.main(path_seg=current_path.joinpath(channel, seg_im_folder),
                                          path_result=current_path.joinpath(channel),
                                          loglevel=args.loglevel)

        if run_tracking:
            # run tracking (we checkpoint after each channel)
            for num, channel in enumerate(config.getlist(identifier, "Channels")):
                # The phase channel is always the first
                if num == 0 and not config.getboolean(identifier, "PhaseSegmentation"):
                    continue

                with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config,
                                       state=f"Tracking_{channel}", identifier=identifier,
                                       copy_path=current_path) as checker:
                    # check to skip
                    checker.check()

                    # track the cells
                    track_cells.main(path=current_path.joinpath(channel),
                                     tracking_class=config.get(identifier, "TrackingClass"),
                                     loglevel=args.loglevel)

        # Cleanup
        for channel in config.getlist(identifier, "Channels"):
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config,
                                   state=f"Cleanup_{channel}", identifier=identifier,
                                   copy_path=current_path) as checker:
                # check to skip
                checker.check()

                # remove everything that the user does not want to keep
                if not config.getboolean(identifier, "KeepCopyOriginal"):
                    # get a list of files to remove
                    file_ext = config.get("General", "FileType")
                    if file_ext == "ome.tif":
                        files = base_path.joinpath(channel).glob(f"*{identifier}*/**/*.ome.tif")
                    else:
                        files = base_path.joinpath(channel).glob(f"*{identifier}*.{file_ext}")

                    # remove the files
                    for file in files:
                        file.unlink(missing_ok=True)
                if not config.getboolean(identifier, "KeepRawImages"):
                    shutil.rmtree(current_path.joinpath(channel, raw_im_folder), ignore_errors=True)
                if not config.getboolean(identifier, "KeepCutoutImages"):
                    shutil.rmtree(current_path.joinpath(channel, cut_im_folder), ignore_errors=True)
                if not config.getboolean(identifier, "KeepSegImagesLabel"):
                    shutil.rmtree(current_path.joinpath(channel, seg_im_folder), ignore_errors=True)
                if not config.getboolean(identifier, "KeepSegImagesBin"):
                    shutil.rmtree(current_path.joinpath(channel, seg_im_bin_folder), ignore_errors=True)
                if not config.getboolean(identifier, "KeepSegImagesTrack"):
                    shutil.rmtree(current_path.joinpath(channel, seg_im_folder, "segmentations_bayesian.h5"),
                                  ignore_errors=True)

        # if we are here, we copy the config file to the identifier
        logger.info(f"Finished with identifier {identifier}, coping settings...")
        config.to_file(current_path)

    logger.info("Done!")


# main routine
if __name__ == "__main__":
    run_module()
