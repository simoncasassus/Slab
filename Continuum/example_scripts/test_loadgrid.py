import sys
import os
import re
import numpy as np
import math
import matplotlib
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
from time import time
#from astropy import constants as const
from astropy.constants import astropyconst20 as const
from pprint import pprint
import dsharp_opac as opacity

# matplotlib.use('Agg')

from numba import jit

griddir = './opac_grids_wb3/'
filetag = ''
kappa_abs_hdulgrid = fits.open(griddir + 'kappa_abs_grid' + filetag + '.fits')
kappa_scat_hdulgrid = fits.open(griddir + 'kappa_scat_grid' + filetag +
                                '.fits')

c_MKS = const.c.value  # DEV
obsfreqs = np.array([c_MKS / 0.3E-2, c_MKS / 0.1E-2])  #DEV

N_freqs = len(kappa_abs_hdulgrid)
print("N_freqs", N_freqs)

hdr = kappa_abs_hdulgrid[0].header
logamaxgrid = hdr['CRVAL2'] + hdr['CDELT2'] * (np.arange(hdr['NAXIS2']) -
                                               (hdr['CRPIX2'] - 1))
logamaxgrid_smallest = hdr['CRVAL2']
logamaxgrid_delta = hdr['CDELT2']
Nlogamaxgrid = hdr['NAXIS2']
qgrid = hdr['CRVAL1'] + hdr['CDELT1'] * (np.arange(hdr['NAXIS1']) -
                                         (hdr['CRPIX1'] - 1))
qgrid_smallest = hdr['CRVAL1']
qgrid_delta = hdr['CDELT1']
Nqgrid = hdr['NAXIS1']

gridshape = list(kappa_abs_hdulgrid[0].data.shape)
gridshape.insert(0, N_freqs)
#xprint("gridshape", gridshape)
kappa_abs_nus_grid = np.zeros(gridshape)
kappa_scat_nus_grid = np.zeros(gridshape)
obsfreqs = np.zeros(N_freqs)
for ifreq in range(N_freqs):
    obsfreqs[ifreq] = kappa_abs_hdulgrid[ifreq].header['freq']
    #print("ifreq", ifreq, self.N_freqs,
    #len(self.kappa_abs_hdulgrid))
    kappa_abs_nus_grid[ifreq, :, :] = kappa_abs_hdulgrid[ifreq].data
    kappa_scat_nus_grid[ifreq, :, :] = kappa_scat_hdulgrid[ifreq].data

print("obsfreqs", obsfreqs)
#print("qgrid",qgrid)
q0 = -3.5
iq = np.argmin(np.fabs(qgrid - q0))
print("iq", iq, "qgrid[iq]", qgrid[iq])

print("kappa_abs_nus_grid.shape", kappa_abs_nus_grid.shape)

import matplotlib.pyplot as plt

ifreq = 2
kabs = 100 * kappa_abs_nus_grid[ifreq, :, iq]

plt.plot(logamaxgrid, kabs)
print(griddir + 'fig_kappa_q-3.5_ifreq' + str(ifreq) + '.png')
plt.savefig(griddir + 'fig_kappa_q-3.5_ifreq' + str(ifreq) + '.png')
