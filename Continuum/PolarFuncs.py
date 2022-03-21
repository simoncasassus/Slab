import os
import numpy as np
import matplotlib
from astropy.io import fits
from copy import deepcopy

import matplotlib.pyplot as plt
from scipy import ndimage
import scipy as sp
import sys

from astropy import constants as const


def polar2cartesian(outcoords, inputshape, origin):
    yindex, xindex = outcoords
    x0, y0 = origin
    nx = inputshape[0] - 1
    ny = inputshape[1] - 1
    x = -float(xindex) + x0
    y = float(yindex) - y0

    theta = np.arctan2(x, y)
    if (theta < 0):
        theta = theta + 2. * np.pi

    thetaindex = (theta * nx / (2. * np.pi))
    rindex = np.sqrt(x**2 + y**2)

    return (rindex, thetaindex)


def polartocart(im_polar):
    nx, ny = im_polar.shape
    im = sp.ndimage.geometric_transform(im_polar,
                                        polar2cartesian,
                                        order=0,
                                        output_shape=(nx, ny),
                                        extra_keywords={
                                            'inputshape':
                                            im_polar.shape,
                                            'origin':
                                            (((nx - 1) / 2), ((ny - 1) / 2))
                                        })

    return im


def cartesian2polar(outcoords, inputshape, origin):

    rindex, thetaindex = outcoords
    x0, y0 = origin
    theta = thetaindex * 2 * np.pi / (inputshape[0] - 1)
    y = rindex * np.cos(theta)
    x = rindex * np.sin(theta)
    ix = -x + x0
    iy = y + y0
    return (iy, ix)


def carttopolar(im):
    nx, ny = im.shape[0], im.shape[1]
    im_polar = sp.ndimage.geometric_transform(im,
                                              cartesian2polar,
                                              order=1,
                                              output_shape=(nx, ny),
                                              extra_keywords={
                                                  'inputshape':
                                                  im.shape,
                                                  'origin': (((nx - 1) / 2),
                                                             ((ny - 1) / 2))
                                              })

    return im_polar
