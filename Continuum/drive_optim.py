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
    outputdir='./output_dev_optim/')

obsfreqs = np.array([100E9, 150E9, 230E9, 345E9])

ZSED = AModelSED.MSED(
    ZSetup,
    Tdust=30.,
    q_dustexpo=-3.5,
    f_grain=1.,  # grain filling factor
    amin=1E-3,  # cm
    amax=1.,  # cm, maximum grain size
    Sigma_g=0.5,  # g/cm2
    gtod_ratio=100.,
    rho0=2.77,  # g/cm3
    N_asizes=40,
    nus=obsfreqs)

ZSED.get_kappa_as()
ZSED.get_taus_and_kappas()
ZSED.get_Inus()

save_mockdata = np.zeros((len(obsfreqs), 3))
save_mockdata[:, 0] = obsfreqs
save_mockdata[:, 1] = ZSED.Inus
save_mockdata[:, 2] = ZSED.Inus * 0.1

np.savetxt(ZSetup.outputdir + 'mockSED.dat', save_mockdata)

obsfreqs_alphas = np.array(
    [100E9, 115E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])

ZSEDalphas = AModelSED.MSED(
    ZSetup,
    Tdust=30.,
    q_dustexpo=-3.5,
    f_grain=1.,  # grain filling factor
    amin=1E-3,  # cm
    amax=1.,  # cm, maximum grain size
    Sigma_g=0.5,  # g/cm2
    gtod_ratio=100.,
    rho0=2.77,  # g/cm3
    N_asizes=400,
    nus=obsfreqs_alphas)

ZSEDalphas.get_kappa_as()
ZSEDalphas.get_taus_and_kappas()
ZSEDalphas.get_Inus()

intraband_accuracy = 0.008
npairs = 4
obsalphas = np.zeros(npairs)
obsnu1s = np.zeros(npairs)
obsnu2s = np.zeros(npairs)
sobsalphas = np.zeros(npairs)
for ipair in range(npairs):
    obsalphas[ipair] = np.log(
        ZSEDalphas.Inus[2 * ipair + 1] / ZSEDalphas.Inus[2 * ipair]) / np.log(
            ZSEDalphas.nus[2 * ipair + 1] / ZSEDalphas.nus[2 * ipair])
    obsnu2s[ipair] = ZSEDalphas.nus[2 * ipair + 1]
    obsnu1s[ipair] = ZSEDalphas.nus[2 * ipair]
    sobsalphas[ipair] = (
        1 / np.log(ZSEDalphas.nus[2 * ipair + 1] /
                   ZSEDalphas.nus[2 * ipair])) * intraband_accuracy

save_mockdata = np.zeros((npairs, 4))
save_mockdata[:, 0] = obsnu1s
save_mockdata[:, 1] = obsnu2s
save_mockdata[:, 2] = obsalphas
save_mockdata[:, 3] = sobsalphas

np.savetxt(ZSetup.outputdir + 'mockalphas.dat', save_mockdata)

# import SEDOptim
