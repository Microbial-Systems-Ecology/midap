import shutil
import sys
from pathlib import Path
from shutil import copyfile, rmtree
import os

import numpy as np
import pandas as pd

from midap.apps import (
    split_frames,
    cut_chamber,
    segment_cells,
    segment_analysis,
    track_cells,
)
from midap.checkpoint import CheckpointManager


def run_mother_machine(config, checkpoint, main_args, logger, restart=False, config_mode = False):
    """
    This function runs the mother machine.
    :param config: The config object to use
    :param checkpoint: The checkpoint object to use
    :param main_args: The args from the main function
    :param logger: The logger object to use
    :param restart: If we are in restart mode
    """

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

                # channel directories (only the raw images here, the rest is per chamber)
                for channel in config.getlist(identifier, "Channels"):
                    current_path.joinpath(channel, raw_im_folder).mkdir(parents=True)

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
                if (
                    config.get(identifier, "Corners") == "None"
                    or config.get(identifier, "Offsets") == "None"
                ):
                    corners = None
                    offsets = None
                else:
                    corners = tuple(
                        [
                            int(corner)
                            for corner in config.getlist(identifier, "Corners")
                        ]
                    )
                    offsets = list(
                        [
                            int(offset)
                            for offset in config.getlist(identifier, "Offsets")
                        ]
                    )
                registration = config.getboolean(identifier, "Registration", fallback=True)
                cut_corners, offsets = cut_chamber.main(
                    channel=paths,
                    cutout_class=config.get(identifier, "CutImgClass"),
                    corners=corners,
                    offsets=offsets,
                    registration=registration,
                )

                # save the corners if necessary
                if corners is None or offsets is None:
                    corners = f"{cut_corners[0]},{cut_corners[1]},{cut_corners[2]},{cut_corners[3]}"
                    offsets = ",".join([str(offset) for offset in offsets])
                    config.set(identifier, "Corners", corners)
                    config.set(identifier, "Offsets", offsets)
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
                    if segmentation_class == "OmniSegmentation":
                        path_model_weights = Path(__file__).parent.parent.joinpath(
                            "model_weights", "model_weights_omni"
                        )
                    elif segmentation_class == "StarDistSegmentation":
                        path_model_weights = Path(__file__).parent.parent.joinpath(
                            "model_weights", "model_weights_legacy"
                        )
                    elif segmentation_class == "UNetSegmentation":
                        path_model_weights = Path(__file__).parent.parent.joinpath(
                            "model_weights", "model_weights_mother_machine"
                        )
                    else:
                        raise ValueError(
                            f"Unknown segmentation class {segmentation_class}"
                        )

                    # point to a chamber for the weights selection
                    path_channel = os.path.join(channel, "chamber_0")
                    weights = segment_cells.main(
                        path_model_weights=path_model_weights,
                        path_pos=current_path,
                        path_channel=path_channel,
                        postprocessing=True,
                        clean_border=False,
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
                        "Preparation of config file is finished. Please follow instructions on "
                        "https://github.com/Microbial-Systems-Ecology/midap/wiki/MIDAP-On-Euler "
                        "to submit your job on the cluster."
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
                offsets = list(
                    [int(offset) for offset in config.getlist(identifier, "Offsets")]
                )
                registration = config.getboolean(identifier, "Registration", fallback=True)
                _ = cut_chamber.main(
                    channel=paths,
                    cutout_class=config.get(identifier, "CutImgClass"),
                    corners=corners,
                    offsets=offsets,
                    registration=registration,
                )

            # run full segmentation (we checkpoint after each channel)
            for num, channel in enumerate(config.getlist(identifier, "Channels")):
                # The phase channel is always the first
                if num == 0 and not config.getboolean(identifier, "PhaseSegmentation"):
                    continue

                # Run the segmentation for all chambers
                offsets = list(
                    [int(offset) for offset in config.getlist(identifier, "Offsets")]
                )
                for chamber in range(len(offsets)):
                    with CheckpointManager(
                        restart=restart,
                        checkpoint=checkpoint,
                        config=config,
                        state=f"SegmentationFull_{channel}_chamber_{chamber}",
                        identifier=identifier,
                        copy_path=current_path,
                    ) as checker:
                        # check to skip
                        checker.check()

                        logger.info(
                            f"Segmenting all frames for {identifier}, channel {channel} and chamber {chamber}..."
                        )

                        # get the current model weight (if defined)
                        model_weights = config.get(
                            identifier, f"ModelWeights_{channel}"
                        )

                        # run the segmentation, the actual path to the weights does not matter anymore since it is selected
                        path_model_weights = Path(__file__).parent.parent.joinpath(
                            "model_weights"
                        )
                        channel_path = os.path.join(channel, f"chamber_{chamber}")
                        _ = segment_cells.main(
                            path_model_weights=path_model_weights,
                            path_pos=current_path,
                            path_channel=channel_path,
                            postprocessing=True,
                            clean_border=False,
                            network_name=model_weights,
                            segmentation_class=config.get(
                                identifier, "SegmentationClass"
                            ),
                            img_threshold=config.getfloat(identifier, "ImgThreshold"),
                        )
                        # analyse the images
                        segment_analysis.main(
                            path_seg=current_path.joinpath(
                                channel, f"chamber_{chamber}", seg_im_folder
                            ),
                            path_result=current_path.joinpath(
                                channel, f"chamber_{chamber}"
                            ),
                            loglevel=main_args.loglevel,
                        )

        if run_tracking:
            # run tracking (we checkpoint after each channel)
            for num, channel in enumerate(config.getlist(identifier, "Channels")):
                # The phase channel is always the first
                if num == 0 and not config.getboolean(identifier, "PhaseSegmentation"):
                    continue

                # Run the segmentation for all chambers
                offsets = list(
                    [int(offset) for offset in config.getlist(identifier, "Offsets")]
                )
                for chamber in range(len(offsets)):
                    with CheckpointManager(
                        restart=restart,
                        checkpoint=checkpoint,
                        config=config,
                        state=f"Tracking_{channel}_chamber_{chamber}",
                        identifier=identifier,
                        copy_path=current_path,
                    ) as checker:
                        # check to skip
                        checker.check()

                        # track the cells
                        track_cells.main(
                            path=current_path.joinpath(channel, f"chamber_{chamber}"),
                            tracking_class=config.get(identifier, "TrackingClass"),
                            loglevel=main_args.loglevel,
                        )

                with CheckpointManager(
                    restart=restart,
                    checkpoint=checkpoint,
                    config=config,
                    state=f"CombineTracking_{channel}",
                    identifier=identifier,
                    copy_path=current_path,
                ) as checker:
                    # check to skip
                    checker.check()

                    # cycle through all chambers and combine the tracking
                    track_dfs = []
                    for chamber in range(len(offsets)):
                        csv_file = sorted(
                            current_path.joinpath(
                                channel, f"chamber_{chamber}", track_folder
                            ).glob("*.csv")
                        )[0]
                        logger.info(f"Reading {csv_file}")
                        df = pd.read_csv(csv_file)

                        # add cell marker
                        if config.get(identifier, "CellMarker") != "none":
                            if config.get(identifier, "CellMarker") in ["top", "both"]:
                                df["top_most"] = 0
                                idx = df.groupby("frame")["x"].idxmin()
                                df["top_most"].iloc[idx] = 1
                            if config.get(identifier, "CellMarker") in [
                                "bottom",
                                "both",
                            ]:
                                df["bottom_most"] = 0
                                idx = df.groupby("frame")["x"].idxmax()
                                df["bottom_most"].iloc[idx] = 1
                            df.to_csv(csv_file, index=False)

                        # add the chamber
                        df["chamber"] = chamber
                        track_dfs.append(df)

                    # combine the dataframes
                    track_df = pd.concat(track_dfs, ignore_index=True)
                    track_df.to_csv(
                        current_path.joinpath(channel, "combined_lineages.csv"),
                        index=False,
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
                # cycle through chambers
                offsets = list(
                    [int(offset) for offset in config.getlist(identifier, "Offsets")]
                )
                for chamber in range(len(offsets)):
                    if not config.getboolean(identifier, "KeepRawImages"):
                        shutil.rmtree(
                            current_path.joinpath(
                                channel, f"chamber_{chamber}", raw_im_folder
                            ),
                            ignore_errors=True,
                        )
                    if not config.getboolean(identifier, "KeepCutoutImages"):
                        shutil.rmtree(
                            current_path.joinpath(
                                channel, f"chamber_{chamber}", cut_im_folder
                            ),
                            ignore_errors=True,
                        )
                    if not config.getboolean(identifier, "KeepCutoutImagesRaw"):
                        shutil.rmtree(
                            current_path.joinpath(
                                channel, f"chamber_{chamber}", cut_im_rawcounts_folder
                            ),
                            ignore_errors=True,
                        )
                    if not config.getboolean(identifier, "KeepSegImagesLabel"):
                        shutil.rmtree(
                            current_path.joinpath(
                                channel, f"chamber_{chamber}", seg_im_folder
                            ),
                            ignore_errors=True,
                        )
                    if not config.getboolean(identifier, "KeepSegImagesBin"):
                        shutil.rmtree(
                            current_path.joinpath(
                                channel, f"chamber_{chamber}", seg_im_bin_folder
                            ),
                            ignore_errors=True,
                        )
                    if not config.getboolean(identifier, "KeepSegImagesTrack"):
                        files = current_path.joinpath(
                            channel, f"chamber_{chamber}", track_folder
                        ).glob(f"segmentations_*.h5")
                        for file in files:
                            file.unlink(missing_ok=True)

        # if we are here, we copy the config file to the identifier
        logger.info(f"Finished with identifier {identifier}, coping settings...")
        config.to_file(current_path)

        logger.info("Done!")

        cut_im_rawcounts_folder = "cut_im_rawcounts"
