import sys
import os
import re
import numpy as np
import matplotlib

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
    Verbose=True,
    PrintChi2s=True,
    ClearOutputDir=False,
    nf=None,
    kf=None,
    GenFigs=True,
    opct_file='opct_mix.txt',
    VerboseInit=False,
    outputdir='./output_grids/')

freqs = np.array([100E9, 115E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])
nfreqs = len(freqs)

amin = 1E-5  #cm
#amin = 1E-4  #cm
Ngrid_amax = 100
smallestamax = 1E-3
largestamax = 10  # cm
#smallestamax = 1E-3
kappa_abs_4interp = np.zeros(freqs,#amaxs = np.logspace(np.log10(amin), np.log10(largestamax), N_asizes)

loga_sizes = (largestamax - amin) * (
    np.arange(N_asizes) / N_asizes) + np.log10(amin)x
a_sizes = 10**(loga_sizes)


ZSED = AModelSED.MSED(
    ZSetup,
    Tdust=20.,
    q_dustexpo=-3.5,
    f_grain=1.,  # grain filling factor
    amin=amin,  # cm
    amax=10.,  # cm, maximum grain size
    Sigma_g=100. * 0.5,  # g/cm2
    gtod_ratio=100.,
    rho0=2.77,  # g/cm3
    N_asizes=1000,
    nus=freqs,
    ExecTimeReport=False,
    GoNumba=True)
ZSED.calcul()
print("ZSED.kappa_as_abs.shape",ZSED.kappa_as_abs.shape)
allkappa_as_abs = ZSED.kappa_as_abs
allkappa_as_scat = ZSED.kappa_as_scat
asizes = ZSED.a_sizes

np.save(ZSED.outputdir+'allkappa_as_abs.npy',allkappa_as_abs)
np.save(ZSED.outputdir+'allkappa_as_scat.npy',allkappa_as_scat)
np.save(ZSED.outputdir+'logasizes.npy',np.log10(asizes))
