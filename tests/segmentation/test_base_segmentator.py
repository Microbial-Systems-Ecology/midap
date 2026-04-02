import pytest
import numpy as np
from midap.segmentation.base_segmentator import SegmentationPredictor
from midap.segmentation.unet_segmentator import UNetSegmentation


def test_base_cutout():
    """
    Tests the SegmentationPredictor abstract base class
    """

    with pytest.raises(TypeError):
        _ = SegmentationPredictor(path_model_weights=None, postprocessing=None)


# Fixtures
##########


@pytest.fixture()
def unet_instance(tmp_path):
    """
    Returns a UNetSegmentation instance (simplest concrete subclass) for testing
    base-class utility methods without any I/O.
    """
    return UNetSegmentation(
        path_model_weights=str(tmp_path), postprocessing=True, div=16, connectivity=1
    )


# Tests for SegmentationPredictor.scale_pixel_vals
###################################################


def test_scale_pixel_vals_range(unet_instance):
    """
    scale_pixel_vals should map the input to [0, 1]
    """
    img = np.array([[0, 128, 255]], dtype=np.uint8)
    scaled = unet_instance.scale_pixel_vals(img)
    assert scaled.min() == pytest.approx(0.0)
    assert scaled.max() == pytest.approx(1.0)


def test_scale_pixel_vals_preserves_shape(unet_instance):
    """
    scale_pixel_vals should not change the array shape
    """
    img = np.random.rand(20, 30)
    assert unet_instance.scale_pixel_vals(img).shape == img.shape


def test_scale_pixel_vals_threshold(unet_instance):
    """
    img_threshold < 1.0 should clip bright pixels before normalisation
    """
    instance = UNetSegmentation(
        path_model_weights=unet_instance.path_model_weights,
        postprocessing=False,
        div=16,
        connectivity=1,
        img_threshold=0.5,
    )
    img = np.array([0.0, 0.4, 0.6, 1.0])
    scaled = instance.scale_pixel_vals(img)
    # Everything above 0.5 * max (= 0.5) is clipped, so max after clipping is 0.5
    assert scaled.max() == pytest.approx(1.0)
    # The value 0.6 and 1.0 should both be clipped to the same ceiling
    assert scaled[-1] == pytest.approx(scaled[-2])


# Tests for SegmentationPredictor.postprocess_seg
##################################################


def test_postprocess_seg_removes_small_objects(unet_instance):
    """
    postprocess_seg should remove stray pixels that are much smaller than the
    average object size.
    """
    seg = np.zeros((60, 60), dtype=int)
    seg[10:50, 10:50] = 1  # large cell (~1600 px)
    seg[0, 0] = 1          # single stray pixel (0.06 % of average area)
    result = unet_instance.postprocess_seg(seg)
    assert result[0, 0] == 0
    assert result[25, 25] > 0


def test_postprocess_seg_preserves_large_objects(unet_instance):
    """
    Large objects must not be removed by postprocess_seg
    """
    seg = np.zeros((60, 60), dtype=int)
    seg[5:30, 5:30] = 1   # cell A
    seg[35:55, 35:55] = 1  # cell B
    result = unet_instance.postprocess_seg(seg)
    assert result[15, 15] > 0
    assert result[45, 45] > 0


def test_postprocess_seg_preserves_shape(unet_instance):
    """
    postprocess_seg must return an array with the same shape as the input
    """
    seg = np.zeros((40, 50), dtype=int)
    seg[5:20, 5:20] = 1
    result = unet_instance.postprocess_seg(seg)
    assert result.shape == seg.shape
