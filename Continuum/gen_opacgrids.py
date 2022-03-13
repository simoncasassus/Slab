import sys
import os
import re
import numpy as np
import matplotlib
from astropy.io import fits

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
from time import time
from astropy import constants as const

from numba import jit

# matplotlib.use('Agg')
HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/Slab/Continuum/'
sys.path.append(include_path)
import AModelSED

c_MKS = const.c.value  # m/s

ZSetup = AModelSED.Setup(
    filetag='',  # False
    PrintChi2s=True,
    ClearOutputDir=False,
    nf=None,
    kf=None,
    GenFigs=True,
    opct_file='opct_mix.txt',
    VerboseInit=False,
    outputdir='./output_grids/')

freqs = np.array([100E9, 130E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])
nfreqs = len(freqs)

Ngrid_amax = 200
smallestamax = 1E-3  #cm
largestamax = 10  # cm
logamax = (np.log10(largestamax) - np.log10(smallestamax)) * (
    np.arange(Ngrid_amax) / Ngrid_amax) + np.log10(smallestamax)
amaxs = 10**(logamax)

Ngrid_q = 100
qmin = -3.99  #cm
qmax = -2  # cm
qs = (qmax - qmin) * (np.arange(Ngrid_q) / Ngrid_q) + qmin

kappa_abs_grid = np.zeros((nfreqs, Ngrid_amax, Ngrid_q))
kappa_scat_grid = np.zeros((nfreqs, Ngrid_amax, Ngrid_q))

ZSED = AModelSED.MSED(
    ZSetup,
    Tdust=20.,
    q_dustexpo=-3.5,
    f_grain=1.,  # grain filling factor
    amin=1E-4,  # cm
    amax=10.,  # cm, maximum grain size
    Sigma_g=100.,  # g/cm2
    gtod_ratio=100.,
    rho0=2.77,  # g/cm3
    N_asizes=1000,
    nus=freqs,
    Verbose=False,
    ExecTimeReport=False,
    GoNumba=True)
ZSED.calcul()

for iamax, aamax in enumerate(amaxs):
    print("iamax", iamax, " < ", Ngrid_amax)
    for iq, aq in enumerate(qs):
        ZSED.amax = aamax
        ZSED.q_dustexpo = aq
        ZSED.calcul(ForcePrep=True)
        kappa_abs = ZSED.kappa_as_abs
        kappa_scat = ZSED.kappa_as_scat
        kappa_abs_grid[:, iamax, iq] = ZSED.kappa_abs
        kappa_scat_grid[:, iamax, iq] = ZSED.kappa_scat

stackhdus_abs = []
stackhdus_scat = []
deltaqs = (qmax - qmin) / Ngrid_q
deltalogamax = (np.log10(largestamax) - np.log10(smallestamax)) * (1 /
                                                                   Ngrid_amax)
hdulabs = fits.HDUList()
hdulscat = fits.HDUList()
for ifreq, afreq in enumerate(freqs):
    print("ifreq ", ifreq, " < ", nfreqs)
    ahdu_abs = fits.PrimaryHDU(kappa_abs_grid[ifreq, :, :])
    ahdu_scat = fits.PrimaryHDU(kappa_scat_grid[ifreq, :, :])
    hdr = ahdu_abs.header
    hdr['freq'] = afreq
    hdr['CDELT2'] = deltalogamax
    hdr['CRVAL2'] = np.log10(smallestamax)
    hdr['CRPIX2'] = 1
    hdr['CTYPE2'] = 'log10(amax)'
    hdr['CDELT1'] = deltaqs
    hdr['CRVAL1'] = qmin
    hdr['CTYPE1'] = 'q_dustexpo'
    hdr['CRPIX1'] = 1
    ahdu_abs.header = hdr
    ahdu_scat.header = hdr
    hdulabs.append(ahdu_abs)
    hdulscat.append(ahdu_scat)
    #stackhdus_abs.append(ahdu_abs)
    #stackhdus_scat.append(ahdu_scat)

#hdul = fits.HDUList(stackhdus_abs)
hdulabs.writeto(ZSetup.outputdir + 'kappa_abs_grid.fits', overwrite=True)
#hdul = fits.HDUList(stackhdus_scat)
hdulscat.writeto(ZSetup.outputdir + 'kappa_scat_grid.fits', overwrite=True)
