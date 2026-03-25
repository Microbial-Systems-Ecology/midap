import argparse
import os
import shutil
import sys
from glob import glob
from pathlib import Path
from shutil import copyfile

from importlib.metadata import version as get_version


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
    parser = argparse.ArgumentParser(
        description=description,
        add_help=True,
        usage="midap [-h] [--restart [RESTART]] [--headless] "
        "[--loglevel LOGLEVEL] [--cpu_only] [--create_config]",
    )
    # This arge is default if the flag is not set, it is const if it is set without arg, and it is the arg if provided
    parser.add_argument(
        "--restart",
        nargs="?",
        default=None,
        const=".",
        type=str,
        help="Restart pipeline from log file. If a path is specified the checkpoint and settings file "
        "will be restored from the path, otherwise the current working directory is searched.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run pipeline in headless mode, ALL parameters have to be set prior in a config file.",
    )
    parser.add_argument(
        "--headless_cluster",
        nargs="*",
        help=(
            "Run pipeline in headless mode on a new machine (i.e. cluster). "
            "Optionally provide a config directory path (default: current working directory), "
            "and optionally a positional identifier"
        ),
    )
    parser.add_argument(
        "--loglevel",
        type=int,
        default=7,
        help="Set logging level of script (0-7), defaults to 7 (max log)",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Sets CUDA_VISIBLE_DEVICES to -1 which will cause most! applications to use CPU only.",
    )
    parser.add_argument(
        "--create_config",
        action="store_true",
        help="If this flag is set, all other arguments will be ignored and a 'settings.ini' config "
        "file is generated in the current working directory. This option is meant generate "
        "config file templates for the '--headless' mode. Note that this will overwrite "
        "if a file already exists.",
    )
    parser.add_argument(
        "--prepare_config_cluster",
        action="store_true",
        help="This option is meant to generate a config file and the output folder structure for a "
        "dataset. The output folder is decompressed and can be uploaded to the cluster to "
        "continue the pipeline in the '--headless' mode. Note that this will overwrite "
        "if a file already exists.",
    )

    parser.add_argument(
        "--cut_data",
        nargs=4,
        metavar=("INPUT_FOLDER", "OUTPUT_FOLDER", "FROM_CUT", "TO_CUT"),
        help="This option will run midap in data cut mode." 
        "Specify parameters for cutting data from a dataset. Requires 4 values: "
        "INPUT_FOLDER (path to input data, must contain .tif or .tiff files only!)," 
        "OUTPUT_FOLDER (path to save output), "
        "FROM_CUT (start index, integer), and" 
        "TO_CUT (end index, integer). "
        "Example usage: --cut_data /path/to/input /path/to/output 10 50. "
        "Extracts for each .tif file frames from position 10 to 50 and saves it in the output folder.",
    )  

    # parsing
    args = parser.parse_args(args)

    # Some constants or conventions
    config_file = "settings.ini"
    check_file = "checkpoints.log"

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
    version = get_version("midap")
    logger.info(f"Running MIDAP version: {version}")

    # imports
    logger.info(f"Importing all dependencies...")
    # Note, some of these dependencies are only used in the individual modules, but we import them here anyway
    # to avoid loading times later
    from midap.checkpoint import Checkpoint, CheckpointManager
    from midap.config import Config
    from midap.apps import (
        download_files,
        init_GUI,
        split_frames,
        cut_chamber,
        segment_cells,
        segment_analysis,
        track_cells,
    )
    from midap.data.reduce_data import filter_data_set
    from midap.main_family_machine import run_family_machine
    from midap.main_mother_machine import run_mother_machine

    logger.info("Done!")

    # Download the files if necessary
    logger.info(f"Checking necessary files...")
    download_files.main(args=[])
    logger.info("Done!")

    # check if we run in data cut mode
    if args.cut_data:
        input_folder, output_folder, from_cut, to_cut = args.cut_data
        try: 
            from_cut = int(from_cut)
            to_cut = int(to_cut)
        except ValueError:
            parser.error("FROM_CUT and TO_CUT must be integers.")
        logger.info("Runing data cut mode...")
        filter_data_set(input_folder, output_folder, from_cut, to_cut)
        return 0

    # create a config file if requested and exit
    if args.create_config:
        config = Config(fname="settings.ini")
        config.to_file(overwrite=True)
        return 0

    # check if we are restarting
    restart = False
    config_mode = False
    if args.restart is not None:
        # we set the restart flag
        restart = True

        # we search for the checkpoint file (recursive subdirectory search)
        checkpoints = glob(os.path.join(args.restart, "**", check_file), recursive=True)

        # exception handling
        if len(checkpoints) == 0:
            raise FileNotFoundError(
                f"No checkpoint found in the restart directory: {args.restart}"
            )
        if len(checkpoints) > 1:
            raise ValueError(
                f"Multiple checkpoints found in the restart directory: {args.restart}"
            )

        # we extract the checkpoing
        prev_checkpoint = Path(checkpoints[0]).absolute()
        logger.info(f"Found checkpoint: {prev_checkpoint}")

        # now we get the corresponding config file
        prev_config = prev_checkpoint.parent.joinpath(config_file)
        if not prev_config.exists():
            raise FileNotFoundError(
                f"No corresponding config file found: {prev_config}"
            )
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
        
    # since we do a full check above if we have headless mode and restart this is an elif
    elif args.headless_cluster is not None:
        
        cluster_args = args.headless_cluster
        path = os.getcwd() if len(cluster_args) == 0 else cluster_args[0]
        position_id = cluster_args[1] if len(cluster_args) > 1 else None
        
        # read the config from the working directory
        logger.info(f"Running in headless mode for cluster, checking config file in folder {args.headless_cluster}")
        if position_id is not None:
            logger.info(f"Running on restricted data position {position_id}")
        config = Config.from_euler_file(path, fname= config_file, full_check=True, position = position_id)
        logger.info(f"Found settings.ini. Updated path in config file to new location...")
        checkpoint = Checkpoint(check_file, cluster_mode=True)
              

    # if selected, we use the GUI to start a session that generates the config file for headless mode on cluster
    elif args.prepare_config_cluster:
        logger.info("Starting up initial GUI for config creation...")
        # start the GUI
        init_GUI.main(config_file=config_file)
        config = Config.from_file(fname=config_file, full_check=False)
        config.fname = os.path.join(Path(config.get("General", "FolderPath")), config_file) #set the config to be created inside the data folder
        logger.info("Starting config creation process...")
        config_mode = True
        checkpoint = Checkpoint(check_file)

    # we are not restarting nor are we in headless mode
    else:
        logger.info("Starting up initial GUI...")
        # start the GUI
        init_GUI.main(config_file=config_file)
        # load in the config and create a checkpoint
        config = Config.from_file(fname=config_file, full_check=False)
        checkpoint = Checkpoint(check_file)

    # run the pipeline
    if config.get("General", "DataType") == "Family_Machine":
        run_family_machine(
            config=config,
            checkpoint=checkpoint,
            main_args=args,
            logger=logger,
            restart=restart,
            config_mode = config_mode
        )
    elif config.get("General", "DataType") == "Mother_Machine":
        run_mother_machine(
            config=config,
            checkpoint=checkpoint,
            main_args=args,
            logger=logger,
            restart=restart,
            config_mode=config_mode
        )
    else:
        raise ValueError(f"Unknown DataType: {config.get('General', 'DataType')}")


# main routine
if __name__ == "__main__":
    run_module()
