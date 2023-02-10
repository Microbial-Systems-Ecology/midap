import logging

import PySimpleGUI as sg
import numpy as np
import re

from glob import glob
from pathlib import Path

from midap.config import Config
from midap.utils import get_logger, get_inheritors

# Get all subclasses for the dropdown menus
###########################################

# get all subclasses from the imcut
from midap.imcut import *
from midap.imcut import base_cutout

imcut_subclasses = [subclass.__name__ for subclass in get_inheritors(base_cutout.CutoutImage)]

# get all subclasses from the segmentations
from midap.segmentation import *
from midap.segmentation import base_segmentator

segmentation_subclasses = [subclass.__name__ for subclass in get_inheritors(base_segmentator.SegmentationPredictor)]

# get all subclasses from the tracking
from midap.tracking import *
from midap.tracking import base_tracking

tracking_subclasses = [subclass.__name__ for subclass in get_inheritors(base_tracking.Tracking)]
tracking_subclasses.remove("DeltaTypeTracking")

# main function of the App
##########################

def main(config_file="settings.ini", loglevel=7):
    """
    The main function of the App
    :param config_file: Name of the config file to save
    :param loglevel: Loglevel of the script.
    """
    logger = get_logger(filepath=__file__, logging_level=loglevel)

    # set layout for GUI
    sg.theme("LightGrey1")
    appFont = ("Arial", 12)
    sg.set_options(font=appFont)

    # First part of the GUI
    common_params = [[sg.Text("Select the input data type: ",
                              key="track_method_text", font="bold")],
                     [sg.DropDown(key="DataType", values=["Family_Machine"], default_value="Family_Machine")],
                     [sg.Text("Choose the target folder: ", key="title_folder_name", font="bold")],
                     [sg.Input(key="folder_name"), sg.FolderBrowse()],
                     [sg.Text("Filetype (e.g. tif, tiff, ome.tif)", key="title_file_type", font="bold")],
                     [sg.Input(key="file_type")],
                     [sg.Text("Identifier of Position/Experiment (e.g. Pos, pos)", key="pos_id",
                              font="bold")],
                     [sg.Input(key="pos")],
                     [sg.Column([[sg.OK(), sg.Cancel()]], key="col_final")]]

    # Finalize the layout
    window = sg.Window("MIDAP: General Setting", common_params).Finalize()

    # Read Params and close window
    event, values = window.read()
    window.close()

    # Return False if we cancel or press "X"
    if event == "Cancel" or event is None:
        logging.critical("GUI was cancelled or unexpectedly closed, exiting...")
        exit(1)

    # readout the params
    general = {"DataType": values["DataType"],
               "FolderPath": values["folder_name"],
               "FileType": values["file_type"],
               "IdentifierName": values["pos"]}

    # Get all the idetifiers
    folder_path = Path(general["FolderPath"])
    if general["FileType"] == "ome.tif":
        files = sorted(folder_path.glob(f"*{general['IdentifierName']}*.export"))
    else:
        files = sorted(folder_path.glob(f"*{general['IdentifierName']}*.{general['FileType']}"))

    unique_identifiers = np.unique([re.search(f"{general['IdentifierName']}\d+", f.name)[0] for f in files])
    if len(unique_identifiers) > 0:
        logger.info(f"Extracted unique identifiers: {unique_identifiers}")
    else:
        logger.critical("No identifiers found in the folder, exiting...")
        exit(1)

    # add as comma separated list
    general["IdentifierFound"] = ",".join(unique_identifiers)

    # init the config
    config = Config(fname=config_file, general=general)

    # We start a GUI for each ID
    for i, id_name in enumerate(unique_identifiers):
        # we create sections for all identifiers
        config.set_id_section(id_name=id_name)

        # the defaults come either from the first section or from the last that we set
        defaults = config[id_name] if i == 0 else config[unique_identifiers[i-1]]

        # Common elements of the next GUI part
        workflow = [[sg.Text("Part of pipeline", justification="center", size=(16, 1))],
                    [sg.T("         "), sg.Radio("Segmentation and Tracking", "RADIO1", key="segm_track",
                                                 default=(defaults["RunOption"] == "both"))],
                    [sg.T("         "), sg.Radio("Segmentation", "RADIO1", key="segm_only",
                                                 default=(defaults["RunOption"] == "segmentation"))],
                    [sg.T("         "), sg.Radio("Tracking", "RADIO1", key="track_only",
                                                 default=(defaults["RunOption"] == "tracking"))]]

        frames = [[sg.Text("Set frame number")],
                  [sg.Input(defaults["StartFrame"], size=(5, 30), key="start_frame"), sg.Text("-")],
                  [sg.Input(defaults["EndFrame"], size=(5, 30), key="end_frame")]]

        # get the default channels
        if defaults["Channels"] == 'None':
            default_ph = ""
            default_ch = ""
        else:
            splits = defaults["Channels"].split(",")
            default_ph = splits[0]
            default_ch = ",".join(splits[1:])

        # Family Machine specific layout
        layout_family_machine = [[sg.Frame("Conditional Run", [[
            sg.Column(workflow, background_color="white"),
            sg.Column(frames)
        ]])],
                                 [sg.Text("Identifier of phase channel (e.g. Phase, PH, ...)", key="phase_check",
                                          font="bold")],
                                 [sg.Input(key="ch1", default_text=default_ph),
                                  sg.Checkbox("Segmentation/Tracking", key="phase_segmentation", font="bold",
                                              default=defaults.getboolean("PhaseSegmentation"))],
                                 [sg.Text("Comma separated list of identifiers of additional \n"
                                          "channels (e.g. eGFP,GFP,YFP,mCheery,TXRED, ...)",
                                          key="channel_1", font="bold")],
                                 [sg.Input(key="ch2", default_text=default_ch)],
                                 [sg.Text("Select how the chamber cutout should be performed: ",
                                          key="imcut_text", font="bold")],
                                 [sg.DropDown(key="imcut", values=imcut_subclasses,
                                              default_value=defaults["CutImgClass"])],
                                 [sg.Text("Select how the cell segmentation should be performed: ",
                                          key="seg_method_text", font="bold")],
                                 [sg.DropDown(key="seg_method", values=segmentation_subclasses,
                                              default_value=defaults["SegmentationClass"])],
                                 [sg.Text("Select how the cell tracking should be performed: ",
                                          key="track_method_text", font="bold")],
                                 [sg.DropDown(key="track_method", values=tracking_subclasses,
                                              default_value=defaults["TrackingClass"])],
                                 [sg.Text("")],
                                 [sg.Text("Preprocessing", font="bold")],
                                 [sg.Checkbox("Deconvolution of images", key="deconv", font="bold",
                                              default=(defaults["Deconvolution"] == "deconv_family_machine"))],
                                 [sg.Text("")],
                                 [sg.Column([[sg.OK(), sg.Cancel()]], key="col_final")]]

        # Finalize the layout
        window = sg.Window(f"Params for '{id_name}' of {unique_identifiers}", layout_family_machine).Finalize()

        # Read Params and close window
        event, values = window.read()
        window.close()

        # Return False if we cancel or press "X"
        if event == "Cancel" or event is None:
            logging.critical("GUI was cancelled or unexpectedly closed, exiting...")
            exit(1)

        # Set the general parameter
        section = {}

        # get all the channels
        channels = values["ch1"]
        # Only an emtpy string is False
        if values["ch2"]:
            channels += f",{values['ch2']}"
        section["Channels"] = channels

        # Read out the Radio Button
        run_options = ["both", "segmentation", "tracking"]
        cond_run = [values["segm_track"], values["segm_only"], values["track_only"]]
        ix_cond = np.where(np.array(cond_run))[0][0]
        section["RunOption"] = run_options[ix_cond]

        # deconv
        if values["deconv"]:
            section["Deconvolution"] = "deconv_family_machine"
        else:
            section["Deconvolution"] = "no_deconv"

        # The remaining generals
        section["StartFrame"] = values["start_frame"]
        section["EndFrame"] = values["end_frame"]
        section["PhaseSegmentation"] = values["phase_segmentation"]

        # The classes
        section["CutImgClass"] = values["imcut"]
        section["SegmentationClass"] = values["seg_method"]
        section["TrackingClass"] = values["track_method"]

        # overwrite the section defaults
        config.read_dict({id_name: section})

    # write to file
    config.to_file(config_file, overwrite=True)

# Run as script
###############

if __name__ == "__main__":
    main()