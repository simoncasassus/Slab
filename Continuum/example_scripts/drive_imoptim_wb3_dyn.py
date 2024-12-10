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
include_path = HOME + '/gitcommon/'
sys.path.append(include_path)

import Slab.Continuum.src.AModelSED as AModelSED
import Slab.Continuum.src.SEDOptim as SEDOptim
import Slab.Continuum.src.ImOptim as ImOptim
import Slab.Continuum.src.GenOpacGridsDSHARP as GenOpacGridsDSHARP
import Slab.Continuum.src.SummaryFigDust as SummaryFigDust
from Slab.Continuum.src.Merit import Merit

datadir = '/home/simon/PDS70snow/slab/proc/'

files_images = [
    datadir + 'B9_z_s.fits',
    datadir + 'B7_z_s.fits',
    datadir + 'B4_z_s.fits',
    datadir + 'B3_z.fits',
]

outputdir = './output_imoptim_wb3_dyn_ncores_optdepth_amax_Sigma_g_regul/'
SED_filename_tag = ''
SingleLOS = None
#SED_filename_tag = '_amaxpeak'
#SingleLOS = [20, 41]  #
#SED_filename_tag = '_Wring'
#SingleLOS = [36, 29]  #
#SED_filename_tag = '_B4peak'
#SingleLOS = [37, 42]  #
ZSetup = AModelSED.Setup(
    filetag='',  # False
    PrintChi2s=True,
    ClearOutputDir=(SingleLOS is None),
    GenFigs=True,
    GoInterp=True,
    griddir='./opac_grids_wb3_dyn/',
    opct_file=
    '/home/simon/gitcommon/Slab/Continuum/opct_mix.txt',  #optical constants
    VerboseInit=False,
    outputdir=outputdir)

hdu_canvas, mfreq_imhdus, omega_beams = ImOptim.loaddata(
    files_images,
    files_specindex=None,
    files_errspecindex=None,
    zoomfactor=8,
    outputdir=outputdir)

omega_beam = omega_beams[0]

obsfreqs = np.array([671E9, 351E9, 145E9, 97.5E9])

rmsnoises = 1E6 * np.array([8.6e-05,
                            2.6e-05,
                            4.7e-06,
                            3*5.4e-06,
                            ])  #rms noise in uJy/beam

fluxcal_accuracy = np.array([0.1, 0.1, 0.05, 0.05])

os.system("rm -rf " + ZSetup.griddir)  # DEV
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
    Sigma_g=1E5,  # g/cm2
    gtod_ratio=100.,
    rho0=2.77,  # g/cm3
    N_asizes=1000,
    nus=obsfreqs)


#Conmpute SED with 
#ZSED.calcul()



"""
Define the merit function - if Regul then pass Lbda* weights 
"""
ZMerit = Merit(ExecTimeReport=False,
               with_specindexdata=False,
               Regul=True,
               LbdaSigma_gRegul=10.,
               MaxOptDepth=1.,
               LbdaTdustRegul=10.,
               LbdaOptDepthRegul=10.,
               LbdaamaxRegul=100.)

domain = [
    ['log(Tdust)', np.log10(15.), [0., 3]],
    #['q_dustexpo', -3.5, [-3.99, -2.]],
    #['f_grain', 1., [0., 1.]],
    ['log(amax)', np.log10(1.), [np.log10(1E-4),
                                 np.log10(100.)]],  #cm
    ['log(Sigma_g)',
     np.log10(10.), [np.log10(1E-5), np.log10(1E3)]]
]  # g/cm2

domain_MCMC = domain
#domain_CG = [['log(Tdust)', np.log10(100.), [0., 3]],
#             ['log(Sigma_g)',
#              np.log10(0.1), [np.log10(1E-5), np.log10(1E3)]]]

nvars = len(domain)
print("nvars: ", nvars)

Reportflags = SingleLOS is not None
OptimM = SEDOptim.OptimM(
    RunConjGrad=False,
    RunSampler=True,
    MCMCresult_UseMedian=False,
    n_cores_MCMC=1,
    n_cores_sampler=1,
    sampler='dynesty',
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
    n_cores_map=100,
    intensity_threshold=[2, 10],  # 10   ifreq, nthres
    files_images=files_images,
    # files_specindex=files_specindex,
    SingleLOS=SingleLOS,
    omega_beams=omega_beams,
    fluxcal_accuracy=fluxcal_accuracy,
    intraband_accuracy=0.008)

if SingleLOS is None:
    fileout = outputdir + 'fig_dustparams.png'

    #errthreshs = [['log(Tdust)', 0.2], ['log(amax)', 1.],
    #              ['log(Sigma_g)', 0.3]]
    errthreshs = None
    fileout = outputdir + 'fig_dustparams.png'
    SummaryFigDust.exec_summary(outputdir,
                                domain,
                                fileout,
                                errthreshs=errthreshs)
