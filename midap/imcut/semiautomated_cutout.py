import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from scipy.signal import find_peaks_cwt
from skimage.registration import phase_cross_correlation

from .base_cutout import CutoutImage

_img = None 
# functions for the pool


def init_pool(img):
    """
    Initializes the pool by setting the image that is used for the cutout
    :param img: The image that is used for the cutout
    """
    global _img
    _img = img


def get_single_shift(x1, x2, y1, y2, offset):
    """
    Calculates the shift for a single chamber
    :param x1: The left x-coordinate of the chamber
    :param x2: The right x-coordinate of the chamber
    :param y1: The lower y-coordinate of the chamber
    :param y2: The upper y-coordinate of the chamber
    :param offset: The offset in x-direction
    :return: A single shift
    """

    global _img

    # get the image
    chamber = _img[y1:y2, x1:x2]
    y_cut = _img[y1:y2]

    # get the shift
    x_dim = x2 - x1
    shift = phase_cross_correlation(
        y_cut[:, offset : offset + x_dim], chamber, normalization=None
    )[0].astype(int)
    return shift


class SemiAutomatedCutout(CutoutImage):
    """
    A class that performs the image cutout for the different channels in interactive mode
    """

    supported_setups = ["Mother_Machine"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the class with given arguments and keyword arguments
        :*args: arguments used to init the parent class
        :**kwargs: keyword arguments used to init the parent class
        """
        # init the super class
        super().__init__(*args, **kwargs)

        # some attributes that will be set later
        self._pool = None
        self._img = None

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

    def get_offsets(self, x1, x2, y1, y2, max_width=100):
        """
        Calculates a list of offsets in x-direction for all detected chambers
        :param x1: The left x-coordinate of the chamber
        :param x2: The right x-coordinate of the chamber
        :param y1: The lower y-coordinate of the chamber
        :param y2: The upper y-coordinate of the chamber
        :param max_width: The maximum width of the chamber in pixels
        """

        # if we are too thick -> do nothing
        x_dim = x2 - x1
        if x_dim > max_width:
            return []

        # get all the valid offsets and calculate the shifts in parallel
        y_cut = self._interactive_img[y1:y2]
        offset = np.arange(0, y_cut.shape[1] - x_dim, 1)
        shifts = self._pool.starmap(
            get_single_shift, [(x1, x2, y1, y2, i) for i in offset]
        )
        shifts = np.stack(shifts, axis=0)

        # we get the minimum shifts by getting the peaks of the min abs
        peaks = find_peaks_cwt(-np.abs(shifts[:, -1]), np.arange(5, 10), min_snr=1)

        # finally the offsets are just the peaks shifted by the x1 coordinate
        offsets = peaks - x1

        # we filter all offsets that are too much to the left (skipped chambers)
        offsets = offsets[offsets > -x_dim // 2]

        return offsets

    def line_select_callback(self, eclick, erelease):
        """
        Line select callback for the RectangleSelector of the <interactive_cutout> routine
        :param eclick: Press event of the mouse
        :param erelease: Release event of the mouse
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # update the plot on the right
        self.ax[1].cla()
        self.ax[1].imshow(self._interactive_img)
        self.ax[1].set_title("Detecting chambers...")
        self.ax[1].set_xticks([])
        self.ax[1].set_yticks([])
        self.ax[1].set_ylim(y2, y1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # get all the offsets
        self.offsets = self.get_offsets(int(x1), int(x2), int(y1), int(y2))

        # update plot again
        self.ax[1].cla()
        self.ax[1].imshow(self._interactive_img)
        self.ax[1].set_title("Current Selection:")
        self.ax[1].set_xticks([])
        self.ax[1].set_yticks([])
        self.ax[1].set_ylim(y2, y1)

        # if we have 3 or more chambers we plot them
        if len(self.offsets) >= 3:
            for num, offset in enumerate(self.offsets):
                c = "r" if num % 2 == 0 else "g"
                self.ax[1].fill_betweenx(
                    [y2, y1], x1 + offset, offset + x2, color=c, alpha=0.5
                )

        plt.draw()

    def interactive_cutout(self, img):
        """
        Generates an interactive plot to select the borders of the chamber
        :param img: The image for the plot as array
        :returns: The corners as (left_x, right_x, lower_y, upper_y)
        """

        # set the image and init the pool witht he image as global variable
        self._interactive_img = img
        self._pool = mp.Pool(4, initializer=init_pool, initargs=(img,))

        # image and selector
        self.fig, self.ax = plt.subplots(1, 2)
        self.ax[0].imshow(img)
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])
        self.ax[0].set_title("Select Region here:")
        rs = RectangleSelector(
            self.ax[0],
            self.line_select_callback,
            drawtype="box",
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
        self.ax[1].set_ylim(y2, y1)
        plt.show()

        # close the pool and extract the corners
        self._pool.close()
        left_x, right_x = rs.corners[0][:2]
        lower_y, upper_y = rs.corners[1][1:3]

        return left_x, right_x, lower_y, upper_y
