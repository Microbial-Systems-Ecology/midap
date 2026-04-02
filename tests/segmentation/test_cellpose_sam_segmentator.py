import os
import tempfile
import numpy as np
import skimage.io as io
import pytest

# Skip the entire module when cellpose v4+ (with cpsam) is not installed
pytest.importorskip("cellpose", reason="cellpose not installed")
from cellpose import models as _cp_models
if "cpsam" not in _cp_models.MODEL_NAMES:
    pytest.skip("cpsam not available in installed cellpose version", allow_module_level=True)

from midap.segmentation.cellpose_sam_segmentator import CellposeSAMSegmentation
from skimage.io import imread
from os import listdir

"""
Note:
Cellpose model loading and inference are bypassed by setting segmentation_method
directly.  The tests verify class initialisation and the inherited run_image_stack
pipeline.
"""

# Fixtures
##########


@pytest.fixture()
def img1():
    """
    Creates a test image: a bright foreground square on a dark background.
    """
    img = np.ones((16, 16))
    return np.pad(img, [[32, 32], [32, 32]], mode="constant", constant_values=0.0)


@pytest.fixture()
def segmentation_instance(monkeypatch, img1):
    """
    Prepares a CellposeSAMSegmentation instance with monkeypatched file I/O.
    """
    tmpdir = tempfile.TemporaryDirectory()

    os.makedirs(os.path.join(tmpdir.name, "cut_im"))
    os.makedirs(os.path.join(tmpdir.name, "seg_im"))
    os.makedirs(os.path.join(tmpdir.name, "seg_im_bin"))

    def fake_list(directory):
        if "cut_im" in directory:
            return ["img1_cut.png", "img2_cut.png", "img3_cut.png"]

    monkeypatch.setattr(os, "listdir", fake_list)
    monkeypatch.setattr(io, "imread", lambda path: img1)

    instance = CellposeSAMSegmentation(
        path_model_weights=tmpdir.name, postprocessing=True, div=16, connectivity=1
    )

    yield instance
    tmpdir.cleanup()


# Tests
#######


def test_gpu_flag_is_bool(segmentation_instance):
    """
    The gpu_available attribute set during __init__ should be a plain bool.
    """
    assert isinstance(segmentation_instance.gpu_available, bool)


def test_run_image_stack(segmentation_instance, img1):
    """
    Tests the run_image_stack pipeline for CellposeSAMSegmentation using a dummy
    segmentation method that returns empty binary masks.
    """
    channel_path = segmentation_instance.path_model_weights

    def dummy_seg(imgs):
        return [np.zeros(img.shape[:2], dtype=int) for img in imgs]

    segmentation_instance.segmentation_method = dummy_seg

    segmentation_instance.run_image_stack(channel_path=channel_path, clean_border=True)

    seg_files = listdir(os.path.join(channel_path, "seg_im"))
    assert len(seg_files) == 3

    for f in seg_files:
        img = imread(os.path.join(channel_path, "seg_im", f))
        assert np.unique(img).size == 1
