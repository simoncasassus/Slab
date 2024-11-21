import os
import numpy as np
import matplotlib
from astropy.io import fits
from copy import deepcopy
from functools import partial
from pprint import pprint

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
include_path = HOME + '/gitcommon/'
sys.path.append(include_path)

import Slab.Continuum.src.AModelSED as AModelSED
import Slab.Continuum.src.SEDOptim as SEDOptim
import Slab.Continuum.src.ImOptim as ImOptim
import Slab.Continuum.src.GenOpacGridsDSHARP as GenOpacGridsDSHARP
import Slab.Continuum.src.SummaryFigDust as SummaryFigDust

datadir = '/home/simon/common/ppdisks/ISO-Oph_2/proc/'

## arrange images in increasing order of wavelength
files_images = [
    datadir + 'B8_r0.8_degrad_to_b3r07_z.fits',
    datadir + 'B6_SB17LB19_r2_degrad_to_b3r07_z.fits',
    datadir + 'B3_r07_zoom.fits',
]

#files_specindex = [
#    datadir + 'specindec_100.fits',
#    datadir + 'specindec_150.fits',
#    datadir + 'specindec_230.fits',
#    datadir + 'specindec_345.fits',
#]
#
#files_errspecindex = [
#    datadir + 'sspecindec_100.fits',
#    datadir + 'sspecindec_150.fits',
#    datadir + 'sspecindec_230.fits',
#    datadir + 'sspecindec_345.fits',
#]

outputdir = './output_imoptim_wb3r07_wfillfactor/'
SED_filename_tag = ''
SingleLOS = None
SingleLOS = [19, 8]  # x, y peak B6
SED_filename_tag = '_peakB6'
#SingleLOS = [9, 18]  # x, y minimum B6 Eastern side
#SED_filename_tag = '_minB6'
ZSetup = AModelSED.Setup(
    filetag='',  # False
    PrintChi2s=True,
    ClearOutputDir=(SingleLOS is None),
    GenFigs=True,
    GoInterp=True,
    griddir='./opac_grids_wb3/',
    opct_file=
    '/home/simon/gitcommon/Slab/Continuum/opct_mix.txt',  #optical constants
    VerboseInit=False,
    outputdir=outputdir)

hdu_canvas, mfreq_imhdus, omega_beams, im_fillfactor = ImOptim.loaddata(
    files_images,
    files_specindex=None,
    files_errspecindex=None,
    file_fillfactor=
    '/home/simon/common/ppdisks/ISO-Oph_2/proc/B6_fillfactor.fits',
    zoomfactor=4,
    outputdir=outputdir)

omega_beam = omega_beams[0]

## arrange SED arrays  in increasing order of wavelength
obsfreqs = np.array([405022769701.0, 224997088629.1, 97502954743.1])
rmsnoises = 1E6 * np.array([2.772e-04, 1.383e-04,  9.895e-05
                            ])  #rms noise in uJy/beam
fluxcal_accuracy = np.array([0.1, 0.05, 0.05])

#c_MKS = const.c.value  # DEV 
#obsfreqs = np.array([c_MKS/0.3E-2, c_MKS/0.1E-2]) #DEV 
#print("obsfreqs",obsfreqs) # DEV
os.system("rm -rf "+ZSetup.griddir) # DEV
if not os.path.isfile(ZSetup.griddir + 'kappa_abs_grid.fits'):
    print("computing  grid for intensity data")
    GenOpacGridsDSHARP.gengrid(obsfreqs, ZSetup, filetag='')
    
# obsnu1s = np.array([100E9, 150E9, 230E9, 345E9])

# obsnu2s = np.array([130E9, 165E9, 245E9, 360E9])

# obsfreqs_alphas = np.array(
#    [100E9, 130E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])

#obsfreqs_alphas = np.array(
#    [100E9, 130E9, 150E9, 180E9, 230E9, 260E9, 345E9, 375E9])

ZData = SEDOptim.Data()
ZData.nus = obsfreqs
#ZData.nu1s_alphas = None
#ZData.nu2s_alphas = None
#ZData.nus_alphas = obsfreqs_alphas
ZData.omega_beam = omega_beam
ZData.rmsnoises = rmsnoises

#ZData.nus_alphas = obsfreqs_alphas

ZSED = AModelSED.MSED(
    ZSetup,
    Tdust=50.,
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

ZMerit = SEDOptim.Merit(ExecTimeReport=False, with_specindexdata=False)
domain = [
    ['log(Tdust)', np.log10(30.), [0., 3]],
    #['q_dustexpo', -3.5, [-3.99, -2.]],
    #['f_grain', 1., [0., 1.]],
    ['log(amax)', np.log10(1.), [np.log10(1E-4), np.log10(100.)]],  #cm
    ['log(Sigma_g)',
     np.log10(30.), [np.log10(1E-5), np.log10(1E3)]]
]  # g/cm2
domain_MCMC = domain
domain_CG = [['log(Tdust)', np.log10(100.), [0., 3]],
             ['log(Sigma_g)',
              np.log10(0.1), [np.log10(1E-5), np.log10(1E3)]]]

nvars = len(domain)
print("nvars: ", nvars)

Reportflags = SingleLOS is not None
OptimM = SEDOptim.OptimM(
    RunConjGrad=False,
    RunMCMC=True,
    MCMCresult_UseMedian=False,
    MCMC_Nit=8000,  #10000,  # 200 MCMC iterations
    nwalkers_pervar=12,  # 10
    burn_in=6000,  #8000,  #100
    n_cores_MCMC=1,
    ChainPlots=Reportflags,
    CornerPlots=Reportflags,
    Report=Reportflags,
    MCMCProgress=Reportflags,
    SummaryPlots=Reportflags,
    filename_tag=SED_filename_tag,
    PhysicalInit=True,
    domain=domain)

ImOptim.exec_imoptim(
    OptimM,
    ZSetup,
    ZData,
    ZSED,
    ZMerit,
    hdu_canvas,
    mfreq_imhdus,
    # mfreq_specindexhdus,
    # mfreq_errspecindexhdus,
    im_fillfactor=im_fillfactor,
    n_cores_map=6,
    intensity_threshold=[1, 14],  # ifreq, nthres
    files_images=files_images,
    # files_specindex=files_specindex,
    SingleLOS=SingleLOS,
    omega_beams=omega_beams,
    fluxcal_accuracy=fluxcal_accuracy,
    intraband_accuracy=0.008)

if SingleLOS is None:
    fileout = outputdir + 'fig_dustparams.png'

    errthreshs = [['log(Tdust)', 0.2], ['log(amax)', 1.],
                  ['log(Sigma_g)', 0.3]]

    fileout = outputdir + 'fig_dustparams.png'
    SummaryFigDust.exec_summary(outputdir,
                                domain,
                                fileout,
                                errthreshs=errthreshs)
