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
    GenFigs=True,
    opct_file='opct_mix.txt',
    VerboseInit=False,
    GoInterp=False,
    #outputdir='./output_dev_optim/')
    #outputdir='output_dev_optim_walphas/')
    #outputdir='output_optim_dev_fluxcal0.01_loosepos_noalphasb/')
    #outputdir='output_optim_dev_fluxcal0.01_loosepos_noRegulcheck/')
    #outputdir='output_optim_dev_fluxcal0.01_loosepos_Regul/')
    outputdir='output_optim_wb9/')
#outputdir='./output_optim_walphas_doublefreqlever_fluxcal1percent/')

obsfreqs = np.array([100E9, 150E9, 230E9, 345E9,694E9])

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
    Tdust=30,  # 
    q_dustexpo=-3.5,  # 
    f_grain=1.,  # grain filling factor
    amin=1E-4,  # cm
    amax=1.,  # 1 cm, maximum grain size
    Sigma_g=30.,  # 50 g/cm2
    gtod_ratio=100.,
    rho0=2.77,  # g/cm3
    N_asizes=1000,
    nus=obsfreqs)

ZSED.calcul()
print("calculated mock SED ")
obsInus = ZSED.Inus.copy()
fluxcal_accuracy = 0.01
AddNoise = False
if AddNoise:
    for ifreq in range(len(obsInus)):
        obsInus[ifreq] += np.random.normal(0.,
                                           obsInus[ifreq] * fluxcal_accuracy,
                                           1)

save_mockdata = np.zeros((len(obsfreqs), 3))
save_mockdata[:, 0] = obsfreqs
save_mockdata[:, 1] = obsInus
save_mockdata[:, 2] = ZSED.Inus * fluxcal_accuracy

np.savetxt(ZSetup.outputdir + 'mockSED.dat', save_mockdata)

#obsfreqs_alphas = np.array(
#    [100E9, 115E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])

obsfreqs_alphas = np.array(
    [100E9, 130E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])

ZSED4alphas = AModelSED.MSED(ZSetup)
ZSED4alphas.copy(ZSED)
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
    sobsalphas[ipair] = (
        1 / np.log(ZSED4alphas.nus[2 * ipair + 1] /
                   ZSED4alphas.nus[2 * ipair])) * intraband_accuracy

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

ZData = SEDOptim.Data(file_obsInus=ZSetup.outputdir + 'mockSED.dat',
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
    #['f_grain', 1., [0., 1.]],
    ['log(amax)', np.log10(1.), [np.log10(1E-3), np.log10(10.)]],  #cm
    ['log(Sigma_g)',
     np.log10(50.), [np.log10(1E-5), np.log10(1E3)]]
]  # g/cm2

domain_Powell = [
    ['log(Tdust)', np.log10(100.), [0., 3]],
    #['q_dustexpo', -3.0, [-3.99, -2.]],
    #['f_grain', 1., [0., 1.]],
    ['log(amax)', np.log10(0.1), [np.log10(1E-3),
                                  np.log10(10.)]],  #cm
    ['log(Sigma_g)',
     np.log10(0.1), [np.log10(1E-5), np.log10(1E3)]]
]  # g/cm2

domain_MCMC = [
    ['log(Tdust)', np.log10(100.), [0., 3]],
    ['q_dustexpo', -3.0, [-3.99, -2.]],
    #['f_grain', 1., [0., 1.]],
    ['log(amax)', np.log10(0.01), [np.log10(1E-3),
                                   np.log10(10.)]],  #cm
    ['log(Sigma_g)',
     np.log10(0.1), [np.log10(1E-5), np.log10(1E3)]]
]  # g/cm2

OptimM = SEDOptim.OptimM(
    RunMCMC=True,
    MCMC_Nit=10000,  #MCMC iterations
    nwalkers_pervar=10,
    burn_in=8000,
    CGmaxiter=False,
    n_cores_MCMC=1,
    domain=domain,
    domain_CG=domain_Powell,
    domain_MCMC=domain_MCMC)

#OptimM.MCMC(ZSetup, ZData, ASED, ZMerit)
#OptimM.MCMC(ZSetup, ZData, ASED, ZMerit)
#print("ASED.Tdust",ASED.Tdust)

OptimM.domain = domain_Powell
#OptimM.ConjGrad(ZSetup, ZData, ASED, ZMerit)

#OptimM.Inherit_Init=True
#print("ASED.Tdust",ASED.Tdust)

OptimM.SummaryPlots = True
OptimM.domain = OptimM.domain_MCMC
[names, mcmc_results, bestparams, modelInus,
 modelalphas] = OptimM.MCMC(ZSetup, ZData, ASED, ZMerit)
print("names", names)
print("mcmc_results", mcmc_results)
print("bestparams", bestparams)
print("modelInus", modelInus)
print("modelalphas", modelalphas)