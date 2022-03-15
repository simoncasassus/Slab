import os
import numpy as np
import matplotlib
from astropy.io import fits
from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
# from time import time,gmtime, strftime
import sys
from multiprocessing import Pool
from tqdm import tqdm

from astropy import constants as const

HOME = os.environ.get('HOME')
include_path = HOME + '/common/python/include'
sys.path.append(include_path)

import PyVtools.Vtools as Vtools
from ImUtils.Resamp import gridding
from ImUtils.Cube2Im import slice0
from Gausssmooth import Gauss_filter

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/Slab/Continuum/'
sys.path.append(include_path)

import AModelSED
import SEDOptim
import ImOptim

files_images = [
    './mockdata/I_100.fits',
    './mockdata/I_150.fits',
    './mockdata/I_230.fits',
    './mockdata/I_345.fits',
    './mockdata/I_694.fits',
]

files_specindex = [
    './mockdata/specindec_100.fits',
    './mockdata/specindec_150.fits',
    './mockdata/specindec_230.fits',
    './mockdata/specindec_345.fits',
]

hdu_canvas, mfreq_imhdus, mfreq_specindexhdus, omega_beams = ImOptim.loaddata(
    files_images, files_specindex, zoomfactor=8)

omega_beam = omega_beams[0]

outputdir = './output_dev_imoptim/'
ZSetup = AModelSED.Setup(
    filetag='',  # False
    PrintChi2s=True,
    ClearOutputDir=False,
    GenFigs=False,
    GoInterp=True,
    opct_file='opct_mix.txt',
    VerboseInit=False,
    outputdir=outputdir)

#obsfreqs = np.array([100E9, 150E9, 230E9, 345E9])
obsfreqs = np.array([100E9, 150E9, 230E9, 345E9, 694E9])

#fluxcal_accuracy = np.array([0.1, 0.1, 0.1, 0.1])
fluxcal_accuracy = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

#obsfreqs_alphas = np.array(
#    [100E9, 115E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])


obsnu1s = np.array([100E9, 150E9, 230E9, 345E9])

obsnu2s = np.array([130E9, 165E9, 245E9, 360E9])

obsfreqs_alphas = np.array(
    [100E9, 130E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])

#obsfreqs_alphas = np.array(
#    [100E9, 130E9, 150E9, 180E9, 230E9, 260E9, 345E9, 375E9])

ZData = SEDOptim.Data()
ZData.nus = obsfreqs
ZData.nu1s_alphas = obsnu1s
ZData.nu2s_alphas = obsnu2s
ZData.nus_alphas = obsfreqs_alphas
ZData.omega_beam = omega_beam

#npairs = len(obsnu1s)
#allnus = np.zeros(int(2 * npairs))
#for ipair in range(npairs):
#    inu1 = int(2 * ipair)
#    inu2 = int(2 * ipair + 1)
#    allnus[inu1] = ZData.nu1s_alphas[ipair]
#    allnus[inu2] = ZData.nu2s_alphas[ipair]
    
ZData.nus_alphas = obsfreqs_alphas

ZSED = AModelSED.MSED(
    ZSetup,
    Tdust=30.,
    q_dustexpo=-3.5,
    f_grain=1.,  # grain filling factor
    amin=1E-4,  # cm
    amax=1.,  # cm, maximum grain size
    Sigma_g=30.,  # g/cm2
    gtod_ratio=100.,
    rho0=2.77,  # g/cm3
    N_asizes=1000,
    nus=obsfreqs)

#ZSED.calcul()

ZMerit = SEDOptim.Merit(ExecTimeReport=False)
domain = [
    ['log(Tdust)', np.log10(30.), [0., 3]],
    ['q_dustexpo', -3.5, [-3.99, -2.]],
    #['f_grain', 1., [0., 1.]],
    ['log(amax)', np.log10(1.), [np.log10(1E-3), np.log10(10.)]],  #cm
    ['log(Sigma_g)',
     np.log10(30.), [np.log10(1E-5), np.log10(1E3)]]
]  # g/cm2
domain_MCMC = domain
domain_CG = [['log(Tdust)', np.log10(100.), [0., 3]],
             ['log(Sigma_g)',
              np.log10(0.1), [np.log10(1E-5), np.log10(1E3)]]]

nvars = len(domain)
print("nvars: ", nvars)
SingleLOS=None
#SingleLOS=[12,12]
#SingleLOS=[16,16]

Reportflags=SingleLOS is not None
OptimM = SEDOptim.OptimM(
    RunConjGrad=True,
    RunMCMC=True,
    MCMC_Nit=300,  # 200 MCMC iterations
    nwalkers_pervar=10,  # 10
    burn_in=200,  #100
    n_cores_MCMC=1,
    ChainPlots=Reportflags,
    CornerPlots=Reportflags,
    Report=Reportflags,
    MCMCProgress=Reportflags,
    SummaryPlots=Reportflags,
    PhysicalInit=True,
    domain=domain,
    domain_CG=domain_CG,
    domain_MCMC=domain)


ImOptim.exec_imoptim(OptimM,
                     ZSetup,
                     ZData,
                     ZSED,
                     ZMerit,
                     hdu_canvas,
                     mfreq_imhdus,
                     mfreq_specindexhdus,
                     n_cores_map=6,
                     files_images=files_images,
                     files_specindex=files_specindex,
                     SingleLOS=SingleLOS,
                     omega_beams=omega_beams,
                     fluxcal_accuracy=fluxcal_accuracy,
                     intraband_accuracy=0.008)
