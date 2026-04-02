import os
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import torch

from cellpose import models

from .base_segmentator import SegmentationPredictor
from ..utils import GUI_selector


class CellposeSAMSegmentation(SegmentationPredictor):
    """
    A class that performs image segmentation using Cellpose SAM (cpsam model)
    """

    supported_setups = ["Family_Machine", "Mother_Machine"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the CellposeSAMSegmentation using the base class init
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the base class init
        """

        # base class init
        super().__init__(*args, **kwargs)

        if platform.processor() == "arm":
            self.gpu_available = torch.mps.is_available()
            self.use_bfloat16 = True
        else:
            self.gpu_available = torch.cuda.is_available()
            self.use_bfloat16 = True

    def set_segmentation_method(self, path_to_cutouts):
        """
        Performs the weight selection for the segmentation network. Sets
        self.segmentation_method to a function that takes a list of input images
        and returns a list of segmentations (binary arrays, 0=background, 1=cell).
        :param path_to_cutouts: The directory in which all the cutout images are
        """

        if self.model_weights is None:
            self.logger.info("Selecting weights...")

            # get the image that is roughly in the middle of the stack
            list_files = np.sort(os.listdir(path_to_cutouts))
            if len(list_files) == 1:
                ix_half = 0
            else:
                ix_half = int(np.ceil(len(list_files) / 2))

            path_img = list_files[ix_half]

            # scale the image for display
            img = self.scale_pixel_vals(
                io.imread(os.path.join(path_to_cutouts, path_img))
            )

            # built-in cpsam model plus any custom models from path_model_weights
            label_dict = {"cpsam": "cpsam"}
            for custom_model in Path(self.path_model_weights).iterdir():
                if (
                    custom_model.is_file()
                    and custom_model.suffix == ""
                    and not custom_model.name.startswith(".")
                ):
                    label_dict[custom_model.name] = custom_model

            figures = []
            for model_name, model_path in label_dict.items():
                self.logger.info("Try model: " + str(model_name))
                if Path(str(model_path)).is_file():
                    model = models.CellposeModel(
                        gpu=self.gpu_available, pretrained_model=str(model_path),
                        use_bfloat16=self.use_bfloat16,
                    )
                else:
                    model = models.CellposeModel(
                        gpu=self.gpu_available, pretrained_model=model_name,
                        use_bfloat16=self.use_bfloat16,
                    )

                try:
                    mask, _, _ = model.eval(
                        img,
                        diameter=None,
                        flow_threshold=0.4,
                        cellprob_threshold=0.0,
                    )
                    seg = (mask > 0).astype(int)
                except Exception as e:
                    self.logger.warning(
                        f"Segmentation with model {model_name} failed: {e}"
                    )
                    seg = np.zeros_like(img, dtype=int)

                # create a plot that can be used as a button image
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img)
                ax.contour(seg, [0.5], colors="r", linewidths=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(model_name)
                figures.append(fig)

            # title for the GUI
            channel = os.path.basename(os.path.dirname(path_to_cutouts))
            # if we just got the chamber folder, we need to go one more up
            if channel.startswith("chamber"):
                channel = os.path.basename(
                    os.path.dirname(os.path.dirname(path_to_cutouts))
                )
            title = f"Segmentation Selection for channel: {channel}"

            # start the GUI
            marked = GUI_selector(
                figures=figures, labels=list(label_dict.keys()), title=title
            )

            # set weights
            self.model_weights = label_dict[marked]

        # load the selected model
        if Path(str(self.model_weights)).is_file():
            model = models.CellposeModel(
                gpu=self.gpu_available, pretrained_model=str(self.model_weights),
                use_bfloat16=self.use_bfloat16,
            )
        else:
            model = models.CellposeModel(
                gpu=self.gpu_available, pretrained_model=self.model_weights,
                use_bfloat16=self.use_bfloat16,
            )

        def seg_method(imgs):
            # scale all images before passing to the model
            imgs = [self.scale_pixel_vals(img) for img in imgs]
            # cellpose v4 returns a list of masks when given a list of images
            try:
                mask_list, _, _ = model.eval(
                    imgs,
                    diameter=None,
                    flow_threshold=0.4,
                    cellprob_threshold=0.0,
                )
            except Exception:
                self.logger.warning("Segmentation failed, returning empty masks!")
                mask_list = [np.zeros(img.shape[:2], dtype=int) for img in imgs]

            # convert labeled masks to binary
            return [(m > 0).astype(int) for m in mask_list]

        # set the segmentation method
        self.segmentation_method = seg_method
