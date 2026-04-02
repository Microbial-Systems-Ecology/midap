import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


from .base_cutout import CutoutImage


class InteractiveCutout(CutoutImage):
    """
    A class that performs the image cutout for the different channels in interactive mode
    """

    supported_setups = ["Family_Machine"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the class with given arguments and keyword arguments
        :*args: arguments used to init the parent class
        :**kwargs: keyword arguments used to init the parent class
        """
        # init the super class
        super().__init__(*args, **kwargs)

    def cut_corners(self, img):
        """
        Given a single aligned image as array, it defines the corners that are used to cut out all images
        :param img: Image to cut as array
        :returns: The corners of the cutout as tuple (left_x, right_x, lower_y, upper_y), where full range of the
                  image, i.e. the limits of the corners, are given by the total number of pixels.
        """

        # interactive cutout of chambers
        corners = self.interactive_cutout(img)
        self.corners_cut = tuple([int(i) for i in corners])

    def line_select_callback(self, eclick, erelease):
        """
        Line select callback for the RectangleSelector of the <interactive_cutout> routine
        :param eclick: Press event of the mouse
        :param erelease: Release event of the mouse
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # update the plot on the right
        self.ax[1].set_xlim(x1, x2)
        self.ax[1].set_ylim(y2, y1)
        self.ax[1].relim()
        plt.draw()

    def interactive_cutout(self, img):
        """
        Generates an interactive plot to select the borders of the chamber
        :param img: The image for the plot as array
        :returns: The corners as (left_x, right_x, lower_y, upper_y)
        """

        # image and selector
        fig, self.ax = plt.subplots(1, 2)
        self.ax[0].imshow(img)
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])
        self.ax[0].set_title("Select Region here:")
        rs = RectangleSelector(
            self.ax[0],
            self.line_select_callback,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        x1, x2, y1, y2 = (
            img.shape[0] // 4,
            3 * img.shape[0] // 4,
            img.shape[1] // 4,
            3 * img.shape[1] // 4,
        )
        rs.extents = (x1, x2, y1, y2)

        # show the zoom
        self.ax[1].imshow(img)
        self.ax[1].set_title("Current Selection:")
        self.ax[1].set_xticks([])
        self.ax[1].set_yticks([])
        self.ax[1].set_xlim(x1, x2)
        self.ax[1].set_ylim(y2, y1)
        plt.show()

        left_x, right_x = rs.corners[0][:2]
        lower_y, upper_y = rs.corners[1][1:3]

        return left_x, right_x, lower_y, upper_y
