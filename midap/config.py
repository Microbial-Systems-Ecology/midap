import os

import git

from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Get all subclasses to check validity of config
################################################

from midap.utils import get_inheritors

# get all subclasses from the imcut
from midap.imcut import *
from midap.imcut import base_cutout

imcut_subclasses = [subclass for subclass in get_inheritors(base_cutout.CutoutImage)]
family_imcut_cls = [
    s.__name__ for s in imcut_subclasses if "Family_Machine" in s.supported_setups
]
mother_imcut_cls = [
    s.__name__ for s in imcut_subclasses if "Mother_Machine" in s.supported_setups
]

# get all subclasses from the segmentations
from midap.segmentation import *
from midap.segmentation import base_segmentator

segmentation_subclasses = [
    subclass for subclass in get_inheritors(base_segmentator.SegmentationPredictor)
]
family_seg_cls = [
    s.__name__
    for s in segmentation_subclasses
    if "Family_Machine" in s.supported_setups
]
mother_seg_cls = [
    s.__name__
    for s in segmentation_subclasses
    if "Mother_Machine" in s.supported_setups
]

# get all subclasses from the tracking
from midap.tracking import *
from midap.tracking import base_tracking

tracking_subclasses = [
    subclass.__name__ for subclass in get_inheritors(base_tracking.Tracking)
]
tracking_subclasses.remove("DeltaTypeTracking")


class Config(ConfigParser):
    """
    A subclass of the ConfigParser defining all values of the MIDAP pipeline.
    """

    def __init__(self, fname: str, general: Optional[dict] = None):
        """
        Initializes the Config of the pipeline, the default values of the sections are updated with the entries
        provided in the dictionary
        :param fname: The name of the file the instance corresponds to
        :param general: A dictionary used for the entries of the General section of the config
        """

        # init the parser
        super().__init__()

        # make all keys case sensitive
        self.optionxform = str

        # save the file_name
        self.fname = fname

        # set the defaults
        self.set_general()

        # update
        if general is not None:
            overwrite = {"General": general}
            self.read_dict(overwrite)

    def set_general(self):
        """
        Sets all values of the Config to the default values
        """

        # get the SHA of the git repo
        try:
            repo = git.Repo(path=Path(__file__).parent, search_parent_directories=True)
            sha = repo.head.object.hexsha
        except Exception:
            sha = "None"

        # set defaults
        self.read_dict(
            {
                "General": {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),
                    "Git hash": sha,
                    "DataType": "Family_Machine",
                    "FolderPath": "None",
                    "FileType": "tif",
                    "IdentifierName": "pos",
                    "IdentifierFound": "None",
                }
            }
        )
        
    def set_path(self, new_path):
        self.set("General","FolderPath",new_path)
        
    def update_model_paths(self):
        """
        Updates the path of any models weights to work properly on a system with new realtive paths
        """
        current_file = Path(__file__).resolve()
        repo_root = current_file.parents[1]
        model_weights_dir = repo_root / "model_weights"

        for section in self.sections():
            for key in self[section]:
                if key.startswith("ModelWeights_"): 
                    original_value = self[section][key]

                    # Heuristic: it's a path if it contains slashes or has a file extension
                    if "/model_weights/" in original_value:
                        # Extract the path after "/model_weights/"
                        post_model_weights = original_value.split("/model_weights/", 1)[1]
                        new_path = model_weights_dir / post_model_weights
                        self[section][key] = str(new_path)

    def validate_general(self):
        """
        Validates the contents of the general section
        :raises: Errors if fields are not valid
        """

        # check the DataType
        allowed_datatype = ["Family_Machine", "Mother_Machine"]
        if self.get("General", "DataType") not in allowed_datatype:
            raise ValueError(f"'DataType' not in {allowed_datatype}")

        # check the paths
        if not (folder_path := Path(self.get("General", "FolderPath"))).exists():
            raise FileNotFoundError(
                f"'FolderPath' not an existing directory: {folder_path}"
            )

        # check if we all Found identifiers are valid
        for identifier in (ids := self.getlist("General", "IdentifierFound")):
            if (id_name := self.get("General", "IdentifierName")) not in identifier:
                raise ValueError(
                    f"Identifier '{id_name}' not in found identifiers: {ids}"
                )

    def set_id_section(self, id_name: str):
        """
        Creates a new section for an identifier with id_name and populates the entries with default values
        :param id_name: Name of the new sections, should be an identifier
        """

        if self.get("General", "DataType") == "Family_Machine":
            self.read_dict(
                {
                    id_name: {
                        "RunOption": "both",
                        "Deconvolution": "no_deconv",
                        "StartFrame": 0,
                        "EndFrame": 10,
                        "PhaseSegmentation": False,
                        "Channels": "None",
                        "CutImgClass": "InteractiveCutout",
                        "Corners": "None",
                        "SegmentationClass": "UNetSegmentation",
                        "TrackingClass": "DeltaV2Tracking",
                        "KeepCopyOriginal": True,
                        "KeepRawImages": True,
                        "KeepCutoutImages": True,
                        "KeepCutoutImagesRaw": True,
                        "KeepSegImagesLabel": True,
                        "KeepSegImagesBin": True,
                        "KeepSegImagesTrack": True,
                        "ImgThreshold": 1.0,
                        "RemoveBorder": False,
                        "FluoChange": False,
                        "Registration": True,
                    }
                }
            )

        elif self.get("General", "DataType") == "Mother_Machine":
            self.read_dict(
                {
                    id_name: {
                        "RunOption": "both",
                        "Deconvolution": "no_deconv",
                        "StartFrame": 0,
                        "EndFrame": 10,
                        "PhaseSegmentation": False,
                        "Channels": "None",
                        "CutImgClass": "SemiAutomatedCutout",
                        "Corners": "None",
                        "Offsets": "None",
                        "SegmentationClass": "OmniSegmentation",
                        "TrackingClass": "STrack",
                        "KeepCopyOriginal": True,
                        "KeepRawImages": True,
                        "KeepCutoutImages": True,
                        "KeepCutoutImagesRaw": True,
                        "KeepSegImagesLabel": True,
                        "KeepSegImagesBin": True,
                        "KeepSegImagesTrack": True,
                        "ImgThreshold": 1.0,
                        "FluoChange": False,
                        "Registration": True,
                    }
                }
            )
        else:
            raise ValueError(f"Unknown DataType: {self.get('General', 'DataType')}")

    def validate_id_section(self, id_name: str, basic=True):
        """
        Validates the content of an ID section.
        :param id_name: Name of the section to check
        :param basic: Only check the parameters that would be set by the initial GUI
        :raises: ValueError if invalid value is found or other Errors accordingly
        """

        # get the machine type
        machine_type = self.get("General", "DataType")

        # run option choices
        allowed_run_options = ["both", "segmentation", "tracking"]
        if self.get(id_name, "RunOption").lower() not in allowed_run_options:
            raise ValueError(f"'RunOption' not in {allowed_run_options}")

        # deconvolution choices
        allowed_deconv = ["no_deconv"]
        if machine_type == "Family_Machine":
            allowed_deconv.append("deconv_family_machine")
        elif machine_type == "Mother_Machine":
            allowed_deconv.append("deconv_well")
        if self.get(id_name, "Deconvolution").lower() not in allowed_deconv:
            raise ValueError(f"'Deconvolution' not in {allowed_deconv}")

        # check the ints
        if (start_frame := self.getint(id_name, "StartFrame")) < 0:
            raise ValueError(
                f"'StartFrame' has to be a positive integer, is: {start_frame}"
            )
        if (
            end_frame := self.getint(id_name, "EndFrame")
        ) < 0 or end_frame <= start_frame:
            raise ValueError(
                f"'EndFrame' has to be a positive integer and larger than 'StartFrame', is: {start_frame}"
            )

        # check the booleans
        _ = self.getboolean(id_name, "Registration", fallback=True)
        _ = self.getboolean(id_name, "PhaseSegmentation")
        _ = self.getboolean(id_name, "KeepCopyOriginal")
        _ = self.getboolean(id_name, "KeepRawImages")
        _ = self.getboolean(id_name, "KeepCutoutImages")
        _ = self.getboolean(id_name, "KeepCutoutImagesRaw")
        _ = self.getboolean(id_name, "KeepSegImagesLabel")
        _ = self.getboolean(id_name, "KeepSegImagesBin")
        _ = self.getboolean(id_name, "KeepSegImagesTrack")
        if machine_type == "Family_Machine":
            _ = self.getboolean(id_name, "RemoveBorder")

        # check the threshold
        if (
            threshold := self.getfloat(id_name, "ImgThreshold")
        ) <= 0.0 or threshold > 1.0:
            raise ValueError(
                f"'ImgThreshold' has to be a float between 0.0 and 1.0, is: {threshold}"
            )

        # check all the classes
        if machine_type == "Family_Machine":
            if self.get(id_name, "CutImgClass") not in family_imcut_cls:
                raise ValueError(f"'Class' of 'CutImg' not in {family_imcut_cls}")
            if self.get(id_name, "SegmentationClass") not in family_seg_cls:
                raise ValueError(f"'Class' of 'Segmentation' not in {family_seg_cls}")
        if machine_type == "Mother_Machine":
            if self.get(id_name, "CutImgClass") not in mother_imcut_cls:
                raise ValueError(f"'Class' of 'CutImg' not in {mother_imcut_cls}")
            if self.get(id_name, "SegmentationClass") not in mother_seg_cls:
                raise ValueError(f"'Class' of 'Segmentation' not in {mother_seg_cls}")
        if self.get(id_name, "TrackingClass") not in tracking_subclasses:
            raise ValueError(f"'Class' of 'Tracking' not in {tracking_subclasses}")

        if not basic:
            # check the corner
            corners = self.get(id_name, "Corners")
            corner_list = self.getlist(id_name, "Corners")
            if len(corner_list) != 4:
                raise ValueError(f"'Corner' is not properly defined: {corners}")
            # check if we have valid integers
            for corner in corner_list:
                _ = int(corner)

            # check the offsets
            if machine_type == "Mother_Machine":
                offsets = self.get(id_name, "Offsets")
                offset_list = self.getlist(id_name, "Offsets")
                if len(offset_list) == 0:
                    raise ValueError(f"'Offsets' is not properly defined: {offsets}")
                # check if we have valid integers
                for offset in offset_list:
                    _ = int(offset)

            # check the model weights
            for channel in self.getlist(id_name, "Channels"):
                if self.get(id_name, "SegmentationClass") in [
                    "UNetSegmentation",
                    "HybridSegmentation",
                ]:
                    model_weights = self.get(id_name, f"ModelWeights_{channel}")
                    model_path = Path(model_weights)
                    if (
                        not (model_path.exists() and model_path.suffix == ".h5")
                        and model_weights != "watershed"
                    ):
                        raise ValueError(
                            f"Invalid 'ModelWeights' for method 'UNetSegmentation': {model_weights}"
                        )
                elif self.get(id_name, "SegmentationClass") == "OmniSegmentation":
                    model_weights = self.get(id_name, f"ModelWeights_{channel}")
                    if model_weights not in [
                        "bact_phase_cp",
                        "bact_fluor_cp",
                        "bact_phase_omni",
                        "bact_fluor_omni",
                    ]:
                        raise ValueError(
                            f"Invalid 'ModelWeights' for method 'OmniSegmentation': {model_weights}"
                        )

    def getlist(self, section, option):
        """
        Return the requested param as a list, i.e. transform from comma separated string to list
        :param section: The section of the parameter
        :param option: The requested option
        :return: A list of strings that was generated from the parameter
        """

        return self.get(section=section, option=option).split(",")

    def to_file(
        self, fname: Union[str, bytes, os.PathLike, None] = None, overwrite=True
    ):
        """
        Write the config into a file
        :param fname: Name of the file to write, defaults to fname attribute. If a directory is specified, the file
                      will be saved in that directory with the same name, if a full path is specified, the full path
                      is used to save the file.
        :param overwrite: Overwrite existing file, defaults to True
        :raises: FileExistsError if overwrite is False and file exists
        """

        # check if we have an argument for the file name
        if fname is not None:
            fname = Path(fname)
            # if we have a dir we add the fname attribute
            if fname.is_dir():
                fname = fname.joinpath(self.fname)
        else:
            fname = Path(self.fname)

        # check
        if not overwrite and fname.exists():
            raise FileExistsError(
                f"File already exists, set overwrite to True to overwrite: {fname}"
            )

        # now we can open a w+ without worrying
        with open(fname, "w+") as f:
            self.write(f)

    @classmethod
    def from_file(cls, fname: Union[str, bytes, os.PathLike], full_check=False):
        """
        Initiates a new instance of the class and overwrites the defaults with contents from a file. The contents read
        from the file will be checked for validity.
        :param fname: The name of the file to read
        :param full_check: If True, all parameters of the file will be checked, otherwise only the initial params.
        :return: An instance of the class
        """

        # get the path
        fname = Path(fname)

        # create a class instance
        if fname.is_file():
            config = Config(fname=fname.name)
        else:
            raise FileNotFoundError(f"File {fname} does not exist!")

        # read the file
        with open(fname, "r") as f:
            config.read_file(f)

        # check validity
        config.validate_general()
        for id_name in config.get("General", "IdentifierFound").split(","):
            config.validate_id_section(id_name=id_name, basic=~full_check)

        # if no error was thrown we return the instance
        return config
    
    @classmethod
    def from_euler_file(cls, path: Union[str, bytes, os.PathLike], fname, full_check=False, position = None):
        """
        Initiates a new instance of the class and overwrites the defaults with contents from a file. The contents read
        from the file. the path in the configu will be updated to the passed folder path. then, the file will be checked for validity.
        :path path: The name of path where the file resides
        :param fname: The name of the file to read
        :param full_check: If True, all parameters of the file will be checked, otherwise only the initial params.
        :return: An instance of the class
        """

        # get the path
        fname = Path(os.path.join(path, fname))

        # create a class instance
        if fname.is_file():
            config = Config(fname=fname.name)
        else:
            raise FileNotFoundError(f"File {fname} does not exist!")

        # read the file
        with open(fname, "r") as f:
            config.read_file(f)
           
        config.set_path(path)
        config.update_model_paths()
        if position is not None:
            config.set("General","IdentifierFound",position)

        # check validity
        config.validate_general()
        for id_name in config.get("General", "IdentifierFound").split(","):
            config.validate_id_section(id_name=id_name, basic=~full_check)

        # if no error was thrown we return the instance
        return config
