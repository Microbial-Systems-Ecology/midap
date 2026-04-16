import os
from pathlib import Path
from typing import Collection, Union, List

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage.segmentation import mark_boundaries
from cellpose_omni import models

from .omni_segmentator import OmniSegmentation
from ..utils import GUI_selector


class OmniSegmentationJupyter(OmniSegmentation):
    """
    A class that performs the image segmentation of the cells using a UNet
    """

    supported_setups = ["Jupyter"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the OmniSegmentationJupyter using the base class init
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the basecalss init
        """

        # base class init
        super().__init__(*args, **kwargs)

    
    def set_segmentation_method_jupyter_all_imgs(self, path_to_cutouts: Union[str, bytes, os.PathLike]):
        """
        Performs the weight selection for the segmentation network. A custom method should use this function to set
        self.segmentation_method to a function that takes an input images and returns a segmentation of the image,
        i.e. an array in the same shape but with values only 0 (no cell) and 1 (cell)
        :param path_to_cutouts: The directory in which all the cutout images are
        """

        # check if we even need to select
        if self.model_weights is None:

            # get the image that is roughly in the middle of the stack
            list_files = np.sort([f for f in os.listdir(path_to_cutouts) if not f.startswith((".", "_"))])

            # scale the image and pad
            imgs = []
            for f in list_files:
                img = self.scale_pixel_vals(io.imread(os.path.join(path_to_cutouts, f)))
                imgs.append(img)

            # Get all the labels
            label_dict = {'bact_phase_cp': 'bact_phase_cp',
                          'bact_fluor_cp': 'bact_fluor_cp',
                          'bact_phase_omni': 'bact_phase_omni',
                          'bact_fluor_omni': 'bact_fluor_omni',}
            for custom_model in self._iter_model_weights():
                label_dict.update({custom_model.name: custom_model})

            self.all_segs_label = {}
            self.all_overl = {}
            for model_name, model_path in label_dict.items():
                print(model_name, model_path)
                if Path(model_path).is_file():
                    model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
                else:
                    model = models.CellposeModel(gpu=True, model_type=model_name)

                # predict, we only need the mask, see omnipose tutorial for the rest of the args
                try:
                    mask, _, _ = model.eval(imgs, channels=[0, 0], rescale=None, mask_threshold=-1,
                                            transparency=True, flow_threshold=0, omni=True, resample=True, verbose=0)
                                    # omni removes axes that are just 1

                    self.seg_bin = (np.array(mask) > 0).astype(int)
                    self.seg_label = mask
                    
                    # now we create an overlay of the image and the segmentation
                    overl = [mark_boundaries(i, s, color=(1, 0, 0)) for i,s in zip(imgs, self.seg_bin)]
                    self.all_overl[model_name] = overl
                    self.all_segs_label[model_name] = self.seg_label

                except ValueError: #in case KNN is throwing an error
                    pass


    def segment_images_jupyter(self, imgs, model_weights):
        # helper function for the seg method
        if Path(model_weights).is_file():
            model = models.CellposeModel(gpu=True, pretrained_model=str(model_weights))
        else:
            model = models.CellposeModel(gpu=True, model_type=model_weights)


        # scale all the images
        imgs = [self.scale_pixel_vals(img) for img in imgs]
        
        # we catch here ValueErrors because omni can fail at masking when there are no cells
        try:
            mask, _, _ = model.eval(imgs, channels=[0, 0], rescale=None, mask_threshold=-1,
                                    transparency=True, flow_threshold=0, omni=True, resample=True, verbose=0)
        except ValueError:
            self.logger.warning('Segmentation failed, returning empty mask!')
            mask = np.zeros((len(imgs), ) + imgs[0].shape, dtype=int)

        self.seg_bin = (np.array(mask) > 0).astype(int)
        self.seg_label = mask

        # add the channel dimension and batch if it was 1

        # set the segmentations method
        #self.segmentation_method = seg_method


