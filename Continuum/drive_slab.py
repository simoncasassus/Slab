import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
# from time import time,gmtime, strftime
import sys

from astropy import constants as const

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/Slab/Continuum/'
sys.path.append(include_path)

import AModelSED 


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
    outputdir='./output_dev/')

#ZSetup.prep()
#ZSetup.load_Opct()

N_freqs = 100
lognu1 = np.log10(30E9)
lognu2 = np.log10(700E9)
lognus = lognu1 + (np.arange(N_freqs) / N_freqs) * (lognu2 - lognu1)
nus = 10**lognus

ZSED = AModelSED.MSED(
    ZSetup,
    Tdust=20.,
    q_dustexpo=-3.5,
    f_grain=1.,  # grain filling factor
    amin=1E-3,  # cm
    amax=1.,  # cm, maximum grain size
    Sigma_g=100.*0.5,  # g/cm2
    gtod_ratio=100.,
    rho0=2.77,  # g/cm3
    N_asizes=40,
    nus=nus)

ZSED.get_kappa_as()
ZSED.get_taus_and_kappas()
ZSED.get_Inus()
ZSED.get_Plot()
