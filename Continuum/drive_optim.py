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
    PrintChi2s=True,
    ClearOutputDir=False,
    GenFigs=True,
    opct_file='opct_mix.txt',
    VerboseInit=False,
    #GoNearNeighbor1D=True,
    GoInterp=False,
    outputdir='output_dev_optim_goInterp_wrms/')
#outputdir='output_dev_optim/')
#outputdir='./output_optim_walphas_doublefreqlever_fluxcal1percent/')

obsfreqs = np.array([100E9, 150E9, 230E9, 345E9, 694E9])
rmsnoises = np.array([9., 9.5, 12, 21.6, 313])  #rms noise in uJy/beam
bmaj=0.04/3600. # 40mas beam
omegabeam = (np.pi/(4.*np.log(2)))*(bmaj*np.pi/180.)**2
rmsnoises /= omegabeam 
rmsnoises *= 1E-6

#Tdust 25.7
#Sigma_g = 98.3
#a_max = 0.103
#dustexpo = -3.49

#ZSED = AModelSED.MSED(
#    ZSetup,
#    Tdust=25.7,  # 25.7
#    q_dustexpo=-3.49,  # -3.5
#    f_grain=1.,  # grain filling factor
#    amin=1E-4,  # cm
#    amax=0.103,  # 1 cm, maximum grain size
#    Sigma_g=98.3,  # 50 g/cm2
#    gtod_ratio=100.,
#    rho0=2.77,  # g/cm3
#    N_asizes=1000,
#    nus=obsfreqs)

ZSED = AModelSED.MSED(
    ZSetup,
    Tdust=31.4,  # 
    q_dustexpo=-2.53,  # 
    #q_dustexpo=-3.5,  # 
    f_grain=1.,  # grain filling factor
    amin=1E-4,  # cm
    amax=0.987,  # 1 cm, maximum grain size
    Sigma_g=97.7,  # 50 g/cm2
    gtod_ratio=100.,
    rho0=2.77,  # g/cm3
    N_asizes=1000,
    nus=obsfreqs)

ZSED.calcul()
print("calculated mock SED ")
obsInus = ZSED.Inus.copy()
fluxcal_accuracy = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
#fluxcal_accuracy = np.array([0.05, 0.05, 0.1, 0.1, 0.2])/2.
sobsInus = np.sqrt((ZSED.Inus * fluxcal_accuracy)**2 + rmsnoises**2)
AddNoise = False
if AddNoise:
    for ifreq in range(len(obsInus)):
        obsInus[ifreq] += np.random.normal(0., sobsInus[ifreq], 1)

save_mockdata = np.zeros((len(obsfreqs), 3))
save_mockdata[:, 0] = obsfreqs
save_mockdata[:, 1] = obsInus
save_mockdata[:, 2] = sobsInus

np.savetxt(ZSetup.outputdir + 'mockSED.dat', save_mockdata)

obsfreqs_alphas = np.array(
    [100E9, 130E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])
rmsnoises_nu1s = np.array([9., 9.5, 12, 21.6])  #rms noise in uJy/beam
rmsnoises_nu2s = np.array([9., 9.5, 12, 21.6])  #rms noise in uJy/beam

#obsfreqs_alphas = np.array(
#    [100E9, 130E9, 150E9, 180E9, 230E9, 260E9, 345E9, 375E9])

ZSED4alphas = AModelSED.MSED(ZSetup)
ZSED4alphas.copy(ZSED)
if ZSED4alphas.GoInterp:
    ZSED4alphas.gridfiletag = '_4alphas'
ZSED4alphas.nus = obsfreqs_alphas
ZSED4alphas.calcul(ForcePrep=True)

intraband_accuracy = 0.008
npairs = 4
obsalphas = np.zeros(npairs)
obsnu1s = np.zeros(npairs)
obsnu2s = np.zeros(npairs)
obsInu1s = np.zeros(npairs)
obsInu2s = np.zeros(npairs)
sobsalphas = np.zeros(npairs)
for ipair in range(npairs):
    obsInu1s[ipair] = ZSED4alphas.Inus[2 * ipair]
    obsInu2s[ipair] = ZSED4alphas.Inus[2 * ipair + 1]
    obsalphas[ipair] = np.log(
        ZSED4alphas.Inus[2 * ipair + 1] /
        ZSED4alphas.Inus[2 * ipair]) / np.log(
            ZSED4alphas.nus[2 * ipair + 1] / ZSED4alphas.nus[2 * ipair])
    obsnu2s[ipair] = ZSED4alphas.nus[2 * ipair + 1]
    obsnu1s[ipair] = ZSED4alphas.nus[2 * ipair]

if rmsnoises_nu1s is not None:  # DEV DEV
    Inu1s = obsInu1s
    Inu2s = Inu1s * (obsnu2s / obsnu1s)**obsalphas
    sigma_2 = np.sqrt((Inu2s * intraband_accuracy)**2 + rmsnoises_nu2s**2)
    sigma_1 = rmsnoises_nu1s
    sobsalphas = (1 / np.log(obsnu2s / obsnu1s)) * np.sqrt(
        (sigma_2 / Inu2s)**2 + (sigma_1 / Inu1s)**2)
else:
    sobsalphas = (1 / np.log(obsnu2s / obsnu1s)) * intraband_accuracy


if AddNoise:
    for ifreq in range(len(obsalphas)):
        obsalphas[ifreq] += np.random.normal(0., sobsalphas[ifreq], 1)

save_mockdata = np.zeros((npairs, 5))
save_mockdata[:, 0] = obsnu1s
save_mockdata[:, 1] = obsnu2s
save_mockdata[:, 2] = obsInu1s
save_mockdata[:, 3] = obsalphas
save_mockdata[:, 4] = sobsalphas

np.savetxt(ZSetup.outputdir + 'mockalphas.dat', save_mockdata)

import SEDOptim

ZData = SEDOptim.Data(file_obsInus=ZSetup.outputdir + 'mockSED.dat', omega_beam=omegabeam,
                      file_obsalphas=ZSetup.outputdir + 'mockalphas.dat')

#ZData = SEDOptim.Data(file_obsInus=ZSetup.outputdir + 'mockSED.dat')

# initial conditions

ASED = AModelSED.MSED(ZSetup)
ASED.copy(ZSED)
ASED.nus = ZData.nus
ASED.calcul(ForcePrep=True)
print("calculated template SED for optim - including kappa grids  ")

ZMerit = SEDOptim.Merit(ExecTimeReport=False, Regul=False)

ASED.ExecTimeReport = False

#print("ZMerit.calc", ZMerit.calcul(ZSetup, ZData, ASED))
#for iiter in range(10):
#    print("ZMerit.calc", ZMerit.calcul(ZSetup, ZData, ASED))

domain = [
    ['log(Tdust)', np.log10(30.), [0., 3]],
    ['q_dustexpo', -3.5, [-3.99, -2.]],
    #['q_dustexpo', -3.5, [-3.6, -3.4]],
    #['f_grain', 1., [0., 1.]],
    ['log(amax)', np.log10(1.), [np.log10(1E-3), np.log10(10.)]],  #cm
    ['log(Sigma_g)',
     np.log10(50.), [np.log10(1E-5), np.log10(1E3)]]
]  # g/cm2


OptimM = SEDOptim.OptimM(
    RunMCMC=True,
    MCMC_Nit=1000,  #MCMC iterations
    nwalkers_pervar=10,
    burn_in=900,
    CGmaxiter=False,
    n_cores_MCMC=1,
    domain=domain,
    PhysicalInit=True)

#OptimM.MCMC(ZSetup, ZData, ASED, ZMerit)
#OptimM.MCMC(ZSetup, ZData, ASED, ZMerit)
#print("ASED.Tdust",ASED.Tdust)

#OptimM.domain = domain_Powell
#OptimM.ConjGrad(ZSetup, ZData, ASED, ZMerit)

#OptimM.Inherit_Init=True
#print("ASED.Tdust",ASED.Tdust)

OptimM.SummaryPlots = True
#OptimM.domain = OptimM.domain_MCMC
[names, mcmc_results, bestparams, modelInus,
 modelalphas] = OptimM.MCMC(ZSetup, ZData, ASED, ZMerit)
print("names", names)
print("mcmc_results", mcmc_results)
print("bestparams", bestparams)
print("modelInus", modelInus)
print("modelalphas", modelalphas)

OptimM.SummaryPlots = True
OptimM.Inherit_Init = True
OptimM.ConjGrad(ZSetup, ZData, ASED, ZMerit)
