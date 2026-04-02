import argparse
import os

# import to get all subclasses
from midap.imcut import *
from midap.imcut import base_cutout

from typing import Optional, Union, Iterable


def main(
    channel: Union[
        str, Iterable[str], bytes, Iterable[bytes], os.PathLike, Iterable[os.PathLike]
    ],
    cutout_class: str,
    corners: Optional[tuple] = None,
    offsets: Optional[list] = None,
    registration: bool = True,
):
    """
    Performs the image cutout and alignment on all images in the paths
    :param channel: A single directory or a list of directories with the images to cut and align
    :param cutout_class: Name of the class used to perform the chamber cutout. Must be defined in a file of
                         midap.imcut and a subclass of midap.imcut.base_cutout.CutoutImage
    :param registration: If True, perform cross-image registration using the first channel. If False,
                         use static corners (no phase channel required).
    """
    # get the right subclass
    class_instance = None
    for subclass in base_cutout.CutoutImage.__subclasses__():
        if subclass.__name__ == cutout_class:
            class_instance = subclass

    # throw an error if we did not find anything
    if class_instance is None:
        raise ValueError(f"Chosen class does not exist: {cutout_class}")

    # cutout classes can only support one type of machine
    if len(class_instance.supported_setups) > 1:
        raise ValueError(
            f"Cutout class {cutout_class} supports more than one machine type!"
        )
    if "Family_Machine" in class_instance.supported_setups:
        cut = class_instance(channel)
        if corners is not None:
            cut.corners_cut = corners
        cut.run_align_cutout(registration=registration)

        return cut.corners_cut
    elif "Mother_Machine" in class_instance.supported_setups:
        cut = class_instance(channel)
        if corners is not None and offsets is not None:
            cut.corners_cut = corners
            cut.offsets = offsets
        cut.run_align_cutout_mother_machine(registration=registration)

        return cut.corners_cut, cut.offsets


# run as main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel",
        type=str,
        nargs="+",
        required=True,
        help="The channels used for to cut out the images.",
    )
    parser.add_argument(
        "--cutout_class",
        type=str,
        required=True,
        help="Name of the class used to perform the chamber cutout. Must be defined in a file of "
        "midap.imcut and a subclass of midap.imcut.base_cutout.CutoutImage",
    )
    args = parser.parse_args()

    # unpack the namespace
    main(**vars(args))
