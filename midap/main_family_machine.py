import shutil
import sys
from pathlib import Path
from shutil import copyfile, rmtree

import numpy as np
import os

from midap.apps import (
    split_frames,
    cut_chamber,
    segment_cells,
    seg_fluo_change_analysis,
    segment_analysis,
    track_cells,
    track_analysis,
)
from midap.checkpoint import CheckpointManager


def run_family_machine(config, checkpoint, main_args, logger, restart=False, config_mode = False):
    """
    This function runs the family machine.
    :param config: The config object to use
    :param checkpoint: The checkpoint object to use
    :param main_args: The args from the main function
    :param logger: The logger object to use
    :param restart: If we are in restart mode
    """

    # Setup
    #######

    # folder names
    raw_im_folder = "raw_im"
    cut_im_folder = "cut_im"
    cut_im_rawcounts_folder = "cut_im_rawcounts"
    seg_im_folder = "seg_im"
    seg_im_bin_folder = "seg_im_bin"
    track_folder = "track_output"

    # get the current base folder
    base_path = Path(config.get("General", "FolderPath"))

    # we cycle through all pos identifiers
    for identifier in config.getlist("General", "IdentifierFound"):
        # read out what we need to do
        run_segmentation = config.get(identifier, "RunOption").lower() in [
            "both",
            "segmentation",
        ]
        # current path of the identifier
        current_path = base_path.joinpath(identifier)

        # stuff we do for the segmentation
        if run_segmentation:
            # setup all the directories
            with CheckpointManager(
                restart=restart,
                checkpoint=checkpoint,
                config=config,
                state="SetupDirs",
                identifier=identifier,
            ) as checker:
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
                    current_path.joinpath(channel, cut_im_rawcounts_folder).mkdir(
                        parents=True
                    )
                    current_path.joinpath(channel, seg_im_folder).mkdir(parents=True)
                    current_path.joinpath(channel, seg_im_bin_folder).mkdir(
                        parents=True
                    )
                    current_path.joinpath(channel, track_folder).mkdir(parents=True)

            # copy the files
            with CheckpointManager(
                restart=restart,
                checkpoint=checkpoint,
                config=config,
                state="CopyFiles",
                identifier=identifier,
                copy_path=current_path,
            ) as checker:
                # check to skip
                checker.check()

                logger.info(f"Copying files for {identifier}")

                # we get all the files in the base bath that match
                file_ext = config.get("General", "FileType")
                if file_ext == "ome.tif":
                    files = base_path.glob(f"*{identifier}_*/**/*.ome.tif")
                else:
                    files = base_path.glob(f"*{identifier}_*.{file_ext}")

                for fname in files:
                    for channel in config.getlist(identifier, "Channels"):
                        if channel in fname.stem:
                            logger.info(f"Copying '{fname.name}'...")
                            copyfile(fname, current_path.joinpath(channel, fname.name))

            # This is just to fill in the config file, i.e. split files 2 frames, get corners, etc
            ######################################################################################

            # split frames
            with CheckpointManager(
                restart=restart,
                checkpoint=checkpoint,
                config=config,
                state="SplitFramesInit",
                identifier=identifier,
                copy_path=current_path,
            ) as checker:
                # check to skip
                checker.check()

                logger.info(f"Splitting test frames for {identifier}")

                # split the frames for all channels
                file_ext = config.get("General", "FileType")
                for channel in config.getlist(identifier, "Channels"):
                    paths = list(current_path.joinpath(channel).glob(f"*.{file_ext}"))
                    if len(paths) == 0:
                        raise FileNotFoundError(
                            f"No file of the type '.{file_ext}' exists for channel {channel}"
                        )
                    if len(paths) > 1:
                        raise FileExistsError(
                            f"More than one file of the type '.{file_ext}' "
                            f"exists for channel {channel}"
                        )

                    # we only get the first frame and the mid frame
                    first_frame = config.getint(identifier, "StartFrame")
                    mid_frame = int(
                        0.5 * (first_frame + config.getint(identifier, "EndFrame"))
                    )
                    frames = np.unique([first_frame, mid_frame])
                    split_frames.main(
                        path=paths[0],
                        save_dir=current_path.joinpath(channel, raw_im_folder),
                        frames=frames,
                        deconv=config.get(identifier, "Deconvolution"),
                        loglevel=main_args.loglevel,
                    )

            # cut chamber and images
            with CheckpointManager(
                restart=restart,
                checkpoint=checkpoint,
                config=config,
                state="CutFramesInit",
                identifier=identifier,
                copy_path=current_path,
            ) as checker:
                # check to skip
                checker.check()

                logger.info(f"Cutting test frames for {identifier}")

                # get the paths
                paths = [
                    current_path.joinpath(channel, raw_im_folder)
                    for channel in config.getlist(identifier, "Channels")
                ]

                # Do the init cutouts
                if config.get(identifier, "Corners") == "None":
                    corners = None
                else:
                    corners = tuple(
                        [
                            int(corner)
                            for corner in config.getlist(identifier, "Corners")
                        ]
                    )
                cut_corners = cut_chamber.main(
                    channel=paths,
                    cutout_class=config.get(identifier, "CutImgClass"),
                    corners=corners,
                )

                # save the corners if necessary
                if corners is None:
                    corners = f"{cut_corners[0]},{cut_corners[1]},{cut_corners[2]},{cut_corners[3]}"
                    config.set(identifier, "Corners", corners)
                    config.to_file()

            # select the networks
            with CheckpointManager(
                restart=restart,
                checkpoint=checkpoint,
                config=config,
                state="SegmentationInit",
                identifier=identifier,
                copy_path=current_path,
            ) as checker:
                # check to skip
                checker.check()

                logger.info(f"Segmenting test frames for {identifier}...")

                # cycle through all channels
                for num, channel in enumerate(config.getlist(identifier, "Channels")):
                    # The phase channel is always the first
                    if num == 0 and not config.getboolean(
                        identifier, "PhaseSegmentation"
                    ):
                        continue

                    # get the current model weight (if defined)
                    model_weights = config.get(
                        identifier, f"ModelWeights_{channel}", fallback=None
                    )

                    # run the selector
                    segmentation_class = config.get(identifier, "SegmentationClass")
                    if segmentation_class == "HybridSegmentation":
                        path_model_weights = Path(__file__).parent.parent.joinpath(
                            "model_weights", "model_weights_hybrid"
                        )
                    elif segmentation_class == "OmniSegmentation":
                        path_model_weights = Path(__file__).parent.parent.joinpath(
                            "model_weights", "model_weights_omni"
                        )
                    elif segmentation_class == "StarDistSegmentation":
                        path_model_weights = Path(__file__).parent.parent.joinpath(
                            "model_weights", "model_weights_stardist"
                        )
                    elif segmentation_class == "CellposeSAMSegmentation":
                        path_model_weights = Path(__file__).parent.parent.joinpath(
                            "model_weights", "model_weights_cellpose_sam"
                        )
                    else:
                        path_model_weights = Path(__file__).parent.parent.joinpath(
                            "model_weights", "model_weights_legacy"
                        )
                    weights = segment_cells.main(
                        path_model_weights=path_model_weights,
                        path_pos=current_path,
                        path_channel=channel,
                        postprocessing=True,
                        clean_border=config.get(identifier, "RemoveBorder"),
                        network_name=model_weights,
                        segmentation_class=segmentation_class,
                        just_select=True,
                        img_threshold=config.getfloat(identifier, "ImgThreshold"),
                    )

                    # save to config
                    if model_weights is None:
                        config.set(identifier, f"ModelWeights_{channel}", weights)
                        config.to_file()
    
    #if we are in config creation mode, we want to exit here                    
    if config_mode:
        config.to_file(overwrite=True)
        logger.info("Cleaning up temporary data:")
        for identifier in config.getlist("General", "IdentifierFound"):
            idenitifer_path = os.path.join(base_path,identifier)
            logger.info(f"Deleting temporary data at: {idenitifer_path}")
            rmtree(idenitifer_path)
        logger.info("Successfully finished config setup for headless mode!")
        return 0
        

    # we cycle through all pos identifiers again to perform all tasks fully
    #######################################################################

    for identifier in config.getlist("General", "IdentifierFound"):
        # read out what we need to do
        run_segmentation = config.get(identifier, "RunOption").lower() in [
            "both",
            "segmentation",
        ]
        run_tracking = config.get(identifier, "RunOption").lower() in [
            "both",
            "tracking",
        ]

        # current path of the identifier
        current_path = base_path.joinpath(identifier)

        # stuff we do for the segmentation
        if run_segmentation:
            # split frames
            with CheckpointManager(
                restart=restart,
                checkpoint=checkpoint,
                config=config,
                state="SplitFramesFull",
                identifier=identifier,
                copy_path=current_path,
            ) as checker:
                # exit if this is only run to prepare config
                if main_args.prepare_config_cluster:
                    sys.exit(
                        "Preparation of config file is finished. Please follow instructions on https://github.com/Microbial-Systems-Ecology/midap/wiki/MIDAP-On-Euler to submit your job on the cluster."
                    )

                # check to skip
                checker.check()

                logger.info(f"Splitting all frames for {identifier}")

                # split the frames for all channels
                file_ext = config.get("General", "FileType")
                for channel in config.getlist(identifier, "Channels"):
                    paths = list(current_path.joinpath(channel).glob(f"*.{file_ext}"))
                    if len(paths) > 1:
                        raise FileExistsError(
                            f"More than one file of the type '.{file_ext}' "
                            f"exists for channel {channel}"
                        )

                    # get all the frames and split
                    frames = np.arange(
                        config.getint(identifier, "StartFrame"),
                        config.getint(identifier, "EndFrame"),
                    )
                    split_frames.main(
                        path=paths[0],
                        save_dir=current_path.joinpath(channel, raw_im_folder),
                        frames=frames,
                        deconv=config.get(identifier, "Deconvolution"),
                        loglevel=main_args.loglevel,
                    )

            # cut chamber and images
            with CheckpointManager(
                restart=restart,
                checkpoint=checkpoint,
                config=config,
                state="CutFramesFull",
                identifier=identifier,
                copy_path=current_path,
            ) as checker:
                # check to skip
                checker.check()

                logger.info(f"Cutting all frames for {identifier}")

                # get the paths
                paths = [
                    current_path.joinpath(channel, raw_im_folder)
                    for channel in config.getlist(identifier, "Channels")
                ]

                # Get the corners and cut
                corners = tuple(
                    [int(corner) for corner in config.getlist(identifier, "Corners")]
                )
                _ = cut_chamber.main(
                    channel=paths,
                    cutout_class=config.get(identifier, "CutImgClass"),
                    corners=corners,
                )

            # run full segmentation (we checkpoint after each channel)
            for num, channel in enumerate(config.getlist(identifier, "Channels")):
                # The phase channel is always the first
                if num == 0 and not config.getboolean(identifier, "PhaseSegmentation"):
                    continue

                with CheckpointManager(
                    restart=restart,
                    checkpoint=checkpoint,
                    config=config,
                    state=f"SegmentationFull_{channel}",
                    identifier=identifier,
                    copy_path=current_path,
                ) as checker:
                    # check to skip
                    checker.check()

                    logger.info(
                        f"Segmenting all frames for {identifier} and channel {channel}..."
                    )

                    # get the current model weight (if defined)
                    model_weights = config.get(identifier, f"ModelWeights_{channel}")

                    # run the segmentation, the actual path to the weights does not matter anymore since it is selected
                    path_model_weights = Path(__file__).parent.parent.joinpath(
                        "model_weights"
                    )
                    _ = segment_cells.main(
                        path_model_weights=path_model_weights,
                        path_pos=current_path,
                        path_channel=channel,
                        postprocessing=True,
                        clean_border=config.get(identifier, "RemoveBorder"),
                        network_name=model_weights,
                        segmentation_class=config.get(identifier, "SegmentationClass"),
                        img_threshold=config.getfloat(identifier, "ImgThreshold"),
                    )
                    # analyse the images
                    segment_analysis.main(
                        path_seg=current_path.joinpath(channel, seg_im_folder),
                        path_result=current_path.joinpath(channel),
                        loglevel=main_args.loglevel,
                    )

            if config.getboolean(identifier, "FluoChange") and not run_tracking:
                logger.info(
                    f"Performs fluo change analysis based on segmentation images..."
                )
                seg_fluo_change_analysis.main(
                    path=current_path,
                    channels=config.getlist(identifier, "Channels"),
                )

        if run_tracking:
            # run tracking (we checkpoint after each channel)
            for num, channel in enumerate(config.getlist(identifier, "Channels")):
                # The phase channel is always the first
                if num == 0 and not config.getboolean(identifier, "PhaseSegmentation"):
                    continue

                with CheckpointManager(
                    restart=restart,
                    checkpoint=checkpoint,
                    config=config,
                    state=f"Tracking_{channel}",
                    identifier=identifier,
                    copy_path=current_path,
                ) as checker:
                    # check to skip
                    checker.check()

                    # track the cells
                    track_cells.main(
                        path=current_path.joinpath(channel),
                        tracking_class=config.get(identifier, "TrackingClass"),
                        loglevel=main_args.loglevel,
                    )

            # Tracking postprocessing
            if config.getboolean(identifier, "FluoChange"):
                track_analysis.main(
                    path=current_path,
                    channels=config.getlist(identifier, "Channels"),
                    tracking_class=config.get(identifier, "TrackingClass"),
                )

        # Cleanup
        for channel in config.getlist(identifier, "Channels"):
            logger.info(f"Cleaning up {identifier} and channel {channel}...")
            with CheckpointManager(
                restart=restart,
                checkpoint=checkpoint,
                config=config,
                state=f"Cleanup_{channel}",
                identifier=identifier,
                copy_path=current_path,
            ) as checker:
                # check to skip
                checker.check()

                # remove everything that the user does not want to keep
                if not config.getboolean(identifier, "KeepCopyOriginal"):
                    # get a list of files to remove
                    file_ext = config.get("General", "FileType")
                    if file_ext == "ome.tif":
                        files = base_path.joinpath(identifier, channel).glob(
                            f"*{identifier}*/**/*.ome.tif"
                        )
                    else:
                        files = base_path.joinpath(identifier, channel).glob(
                            f"*{identifier}*.{file_ext}"
                        )

                    # remove the files
                    for file in files:
                        file.unlink(missing_ok=True)
                if not config.getboolean(identifier, "KeepRawImages"):
                    shutil.rmtree(
                        current_path.joinpath(channel, raw_im_folder),
                        ignore_errors=True,
                    )
                if not config.getboolean(identifier, "KeepCutoutImages"):
                    shutil.rmtree(
                        current_path.joinpath(channel, cut_im_folder),
                        ignore_errors=True,
                    )
                if not config.getboolean(identifier, "KeepCutoutImagesRaw"):
                    shutil.rmtree(
                        current_path.joinpath(channel, cut_im_rawcounts_folder),
                        ignore_errors=True,
                    )
                if not config.getboolean(identifier, "KeepSegImagesLabel"):
                    shutil.rmtree(
                        current_path.joinpath(channel, seg_im_folder),
                        ignore_errors=True,
                    )
                if not config.getboolean(identifier, "KeepSegImagesBin"):
                    shutil.rmtree(
                        current_path.joinpath(channel, seg_im_bin_folder),
                        ignore_errors=True,
                    )
                if not config.getboolean(identifier, "KeepSegImagesTrack"):
                    files = current_path.joinpath(channel, track_folder).glob(
                        f"segmentations_*.h5"
                    )
                    for file in files:
                        file.unlink(missing_ok=True)

        # if we are here, we copy the config file to the identifier
        logger.info(f"Finished with identifier {identifier}, coping settings...")
        config.to_file(current_path)

    logger.info("Done!")
