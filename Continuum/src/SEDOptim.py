import sys
import os
import re
import numpy as np
import matplotlib
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize as op
import cmath as cma

from pprint import pprint
import emcee
import corner

from multiprocessing import Pool

from astropy import constants as const
#from astropy.constants import astropyconst20 as const

from .AModelSED import MSED
from .Likelihood import logL, lnlike
from .pre_post_optim import initoptim
from .drive_dynesty import exec_dynesty
from .drive_emcee import exec_emcee
from .drive_scipy_optimize import exec_ConjGrad

c_MKS = const.c.value  # m/s
k_B = const.k_B.value

#HOME = os.environ.get('HOME')
#include_path = HOME + '/gitcommon/'
#sys.path.append(include_path)

if not sys.warnoptions:
    import os, warnings
    #warnings.simplefilter("default") # Change the filter in this process
    warnings.simplefilter("ignore")  # Change the filter in this process
    #os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

#def lnlike_4pool(x_free, parnames, ZSetup, ZData, ASED, ASED4alphas, ZMerit):
#    process = multiprocessing.current_process()
#    pid = process.ident
#    iGPU = pid % ZSetup.ngpus
#
#    return lnlike(x_free, parnames, ZSetup, ZData, ASED, ASED4alphas, ZMerit)


class Data():
    def __init__(
            self,
            file_obsInus='',
            file_obsalphas=False,  # ''
            VerboseInit=False,
            ######################################################################
            nus=None,
            Inus=None,
            rmsnoises=None,
            rmsnoises_nu1s=None,
            rmsnoises_nu2s=None,
            sInus=None,  # standard deviation error 
            nu1s_alphas=None,
            Inu1s=None,
            omega_beam=(np.pi / (4. * np.log(2))) *
        (0.040 * np.pi / (180. * 3600.))**2,  # sr - example is for C10 B3 beam
            nu2s_alphas=None,
            nus_alphas=None,
            alphas=None,
            salphas=None  # standard deviation error 
    ):
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            if VerboseInit:
                print("Data> setting ", a_attribute, " to ",
                      initlocals[a_attribute])
            setattr(self, a_attribute, initlocals[a_attribute])

        self.prep()

    def prep(self):
        if self.file_obsInus:
            self.nus, self.Inus, self.sInus = np.loadtxt(self.file_obsInus,
                                                         unpack=True)
        if self.file_obsalphas:
            self.nu1s_alphas, self.nu2s_alphas, self.Inu1s, self.alphas, self.salphas = np.loadtxt(
                self.file_obsalphas, unpack=True)
            npairs = len(self.nu1s_alphas)
            allnus = np.zeros(int(2 * npairs))
            for ipair in range(npairs):
                inu1 = int(2 * ipair)
                inu2 = int(2 * ipair + 1)
                allnus[inu1] = self.nu1s_alphas[ipair]
                allnus[inu2] = self.nu2s_alphas[ipair]
            self.nus_alphas = allnus

    #def copy(self, AnotherData):
    #    self.__dict__.update(AnotherData.__dict__)

    #def copy(self, AnotherData):
    #    #self.__dict__.update(AnotherSED.__dict__)
    #    attributes_source = AnotherData.__dict__
    #    for anattribute in attributes_source.keys():
    #        value_source = getattr(AnotherData, anattribute)
    #        attributes_source[anattribute] = deepcopy(value_source)
    #        #if isinstance(value_source, np.ndarray):
    #        #    #print("found ndarray ", anattribute)
    #        #    attributes_source[anattribute] = value_source.copy()
    #    self.__dict__.update(attributes_source)


class OptimM():
    def __init__(
            self,
            RunSampler=False,
            RunMCMC=None,
            MCMCresult_UseMedian=False,
            RunConjGrad=True,
            CGmaxiter=100,
            n_cores_sampler=1,
            ######################################################################
            sampler='emcee',
            MCMC_Nit=100,  #MCMC iterations
            nwalkers_pervar=10,
            burn_in=50,
            mcmcinitball_relrange=0.1,
            n_cores_MCMC=None,
            ChainPlots=True,
            ######################################################################
            #sampler='dynesty',
            maxiter=10000,  #MCMC iterations
            nlive_pervar=100,
            ######################################################################
            CornerPlots=True,  # TriangleFile='cornerplot.png',
            PrintChi2s=False,
            Report=True,
            SummaryPlots=True,
            label4SED=None,  # string
            filename_tag='',
            trueparams=False,  # string 
            domain=[],
            domain_CG=[],
            domain_MCMC=[],
            Inherit_Init=False,  # loads init conditions from ASED 
            MCMCProgress=True,
            PhysicalInit=True,  # use physically motivated initial conditions
            summarySED_xaxisscale='log',
            summarySED_yaxisscale='log',
            summaryWithSEDchains=False,
            ######################################################################
            mcmc_results=[],
            mcmc_results_0=[],
            mcmc_bestparams=[]):

        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            print("setting ", a_attribute, " to ", initlocals[a_attribute])
            setattr(self, a_attribute, initlocals[a_attribute])

        if RunMCMC is None:
            self.RunMCMC = RunSampler
        if n_cores_MCMC is None:
            self.n_cores_MCMC = n_cores_sampler

    def MCMC(self, ZSetup, ZData, ASED, ZMerit):
        [names, mcmc_results, bestparams, modelInus, modelalphas,
         achi2] = exec_emcee(self, ZSetup, ZData, ASED, ZMerit)
        return [names, mcmc_results, bestparams, modelInus, modelalphas, achi2]

    def dynesty(self, ZSetup, ZData, ASED, ZMerit):
        [names, mcmc_results, bestparams, modelInus, modelalphas,
         achi2] = exec_dynesty(self, ZSetup, ZData, ASED, ZMerit)
        return [names, mcmc_results, bestparams, modelInus, modelalphas, achi2]

    def ConjGrad(self, ZSetup, ZData, ASED, ZMerit):
        [names, result_ml, modelInus, modelalphas,
         achi2] = exec_ConjGrad(self, ZSetup, ZData, ASED, ZMerit)
        return [names, result_ml, modelInus, modelalphas, achi2]
