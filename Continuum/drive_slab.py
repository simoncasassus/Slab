import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
# from time import time,gmtime, strftime
import sys
from time import time
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
    GoInterp=True,
    opct_file='opct_mix.txt',
    VerboseInit=False,
    outputdir='./output_grids/')

#ZSetup.prep()
#ZSetup.load_Opct()

N_freqs = 100
lognu1 = np.log10(30E9)
lognu2 = np.log10(700E9)
lognus = lognu1 + (np.arange(N_freqs) / N_freqs) * (lognu2 - lognu1)
nus = 10**lognus
freqs = np.array([100E9, 115E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])
nus = freqs
ZSED = AModelSED.MSED(
    ZSetup,
    Tdust=20.,
    q_dustexpo=-3.5,
    f_grain=1.,  # grain filling factor
    amin=1E-4,  # cm
    amax=1.,  # cm, maximum grain size
    Sigma_g=100. * 0.5,  # g/cm2
    gtod_ratio=100.,
    rho0=2.77,  # g/cm3
    N_asizes=1000,
    nus=nus,
    ExecTimeReport=False,
    GoNumba=True)

for iter in range(10):
    print("iter", iter)
    time_1 = time()
    ZSED.calcul()
    time_2 = time()
    print("total execution time:", time_2 - time_1, "s")

time_1 = time()
for iter in range(100):
    ZSED.calcul()
time_2 = time()
print("100 batch execution time:", time_2 - time_1, "s")



ZSED.ExecTimeReport=True
time_1 = time()
ZSED.calcul()
time_2 = time()
print("total execution time:", time_2 - time_1, "s")

# ZSED.get_Plot()
