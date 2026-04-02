import os
import tempfile
import numpy as np
import skimage.io as io
import pytest

from midap.segmentation.unet_segmentator import UNetSegmentation
from skimage.io import imread
from pytest import fixture
from pathlib import Path
from os import listdir

"""
Note:
Most methods tested here are actually from the abstract base class SegmentationPredictor. The interactive 
method of the weight selection is actually monkeypatched in.
"""

# Fixtures
##########


@fixture()
def img1():
    """
    Creates a test image
    :return: A test image used as base image
    """
    # define the images
    img = np.ones((16, 16))

    # we pad the image with 0s to have a bigger image
    pad_img = np.pad(
        img, pad_width=[[32, 32], [32, 32]], mode="constant", constant_values=0.0
    )

    return pad_img


@fixture()
def segmentation_instance(monkeypatch, img1):
    """
    This fixture prepares the InteractiveCutout class including monkey patching for the image read
    :param monkeypatch: The monkypatch fixture from pytest to override methods
    :param img1: A test image fixture used as base image to segment
    :return: A CutoutImage instance
    """

    # create a temp directory
    tmpdir = tempfile.TemporaryDirectory()

    # directories for the read and write
    path_cut = os.path.join(tmpdir.name, "cut_im")
    os.makedirs(path_cut)
    path_seg = os.path.join(tmpdir.name, "seg_im")
    os.makedirs(path_seg)
    path_seg = os.path.join(tmpdir.name, "seg_im_bin")
    os.makedirs(path_seg)

    def fake_list(directory):
        """
        Monkeypatch for os.listdir
        :param directory: path of the direcotry
        :return: A list of images depending on the input
        """

        # we return different images depending on the channel
        if "cut_im" in directory:
            return ["img1_cut.png", "img2_cut.png", "img3_cut.png"]

    # patch
    monkeypatch.setattr(os, "listdir", fake_list)

    def fake_load(path):
        """
        This is a monkeypatch for io.imread
        :param path: Path of the image to load
        :return: A loaded image
        """

        return img1

    # patch
    monkeypatch.setattr(io, "imread", fake_load)

    # get the instance
    unet = UNetSegmentation(
        path_model_weights=tmpdir.name, postprocessing=True, div=16, connectivity=1
    )

    yield unet

    # clean up
    tmpdir.cleanup()


# Tests
#######


def test_run_image_stack(segmentation_instance):
    """
    Tests the run_image_stack method of the SegmentationPredictor class and all other class functions in the process
    :param segmentation_instance: A pytest fixture preparing and monkeypatching the class instance for testing
    """

    # we read out the path model weights (because we stored the tempdir in there)
    channel_path = segmentation_instance.path_model_weights

    # we set the segmentation method to watershed for starters
    segmentation_instance.model_weights = "watershed"

    # run the stack
    segmentation_instance.run_image_stack(channel_path=channel_path, clean_border=True)

    # we check that we got three files (we use the directly imported methods because the others were monkeypatched)
    seg_files = listdir(os.path.join(channel_path, "seg_im"))
    assert len(seg_files) == 3

    # we check that was labeled correctly
    for f in seg_files:
        fpath = os.path.join(channel_path, "seg_im", f)
        img = imread(fpath)
        # watershed fails for such an image because it classifies the background that gets removed
        assert np.unique(img).size == 1

    # path to actual model weights
    weight_path = Path(__file__).absolute().parent.parent.parent
    weight_path = weight_path.joinpath(
        "model_weights",
        "model_weights_family_mother_machine",
        "model_weights_CB15-WT.h5",
    )
    segmentation_instance.model_weights = str(weight_path)

    # run the stack
    segmentation_instance.run_image_stack(channel_path=channel_path, clean_border=True)

    # we check that was labeled correctly
    for f in seg_files:
        fpath = os.path.join(channel_path, "seg_im", f)
        img = imread(fpath)
        # same as for watershed it fails now because of border cell removal
        assert np.unique(img).size == 1


# Fixture for utility-method tests that does not need file I/O monkeypatching
@fixture()
def unet_plain(tmp_path):
    return UNetSegmentation(
        path_model_weights=str(tmp_path), postprocessing=False, div=16, connectivity=1
    )


# Tests for UNetSegmentation utility methods
############################################


def test_pad_image_divisible(unet_plain):
    """
    pad_image should return a shape whose spatial dims are multiples of div=16
    """
    img = np.random.rand(30, 45)
    padded = unet_plain.pad_image(img)
    assert padded.shape[1] % 16 == 0
    assert padded.shape[2] % 16 == 0


def test_pad_image_batch_channel_dims(unet_plain):
    """
    pad_image should add a batch dim (axis 0) and a channel dim (axis -1)
    """
    img = np.random.rand(20, 25)
    padded = unet_plain.pad_image(img)
    assert padded.ndim == 4
    assert padded.shape[0] == 1
    assert padded.shape[-1] == 1


def test_undo_padding_restores_shape(unet_plain):
    """
    undo_padding should exactly reverse pad_image
    """
    img = np.random.rand(37, 53)
    padded = unet_plain.pad_image(img)
    restored = unet_plain.undo_padding(padded)
    assert restored.shape == img.shape


def test_undo_padding_preserves_values(unet_plain):
    """
    Values in the un-padded region should be identical to the original image
    """
    img = np.random.rand(20, 20)
    padded = unet_plain.pad_image(img)
    restored = unet_plain.undo_padding(padded)
    np.testing.assert_array_almost_equal(restored, img)


def test_segment_region_based_output_shape(unet_plain):
    """
    segment_region_based should return an array with the same spatial shape as the input
    """
    img = np.random.rand(40, 40)
    seg = unet_plain.segment_region_based(img, min_val=0.16, max_val=0.19)
    assert seg.shape == img.shape


def test_segment_region_based_binary(unet_plain):
    """
    segment_region_based should return only 0 and 1 values
    """
    img = np.random.rand(40, 40)
    seg = unet_plain.segment_region_based(img, min_val=0.16, max_val=0.19)
    assert set(np.unique(seg)).issubset({0, 1})


def test_seg_method_watershed_output_count(segmentation_instance, img1):
    """
    seg_method_watershed should return one segmentation per input image
    """
    imgs = [img1, img1, img1]
    segs = segmentation_instance.seg_method_watershed(imgs)
    assert len(segs) == 3


def test_seg_method_watershed_output_shape(segmentation_instance, img1):
    """
    Each segmentation from seg_method_watershed should have the same spatial shape
    as the corresponding input image
    """
    imgs = [img1]
    segs = segmentation_instance.seg_method_watershed(imgs)
    assert segs[0].shape == img1.shape
