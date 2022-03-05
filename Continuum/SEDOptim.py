import sys
import os
import re
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
# from time import time,gmtime, strftime

from astropy import constants as const

c_MKS = const.c.value  # m/s

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/Slab/Continuum/'
sys.path.append(include_path)

import AModelSED 

class Data():
    def __init__(
            self,
            files_obsInus='',
            files_obsalphas=False, # ''
            VerboseInit=False,
            nus=None,
            Inus=None,
            sInus=None, # standard deviation error 
            nus_alphas=None,
            alphas=None,
            salphas=None # standard deviation error 
            ):
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            if VerboseInit:
                print("Data> setting ", a_attribute, " to ",
                      initlocals[a_attribute])
            setattr(self, a_attribute, initlocals[a_attribute])

    def prep(self, ZSetup):


class Merit(AModelSED.Setup,Data):
    def __init__(
        self,
        VerboseInit=False,
        ExecTimeReport=False,
    ):
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            if VerboseInit:
                print("Merit> setting ", a_attribute, " to ",
                      initlocals[a_attribute])
            setattr(self, a_attribute, initlocals[a_attribute])
            
    self.__dict__.update(ASED.__dict__)

    def calc(self):
        if self.ExecTimeReport:
            time_start = time()

        chi2 = logL(self)

        if self.ExecTimeReport:
            time_end = time()
            print("time for Likelihood evaluation: ", time_end-time_start," s")
            
        return chi2


class OptimModel():

    def __init__(
            self,
            RunMCMC=False,
            MCMC_Nit=100,  #MCMC iterations
            nwalkers_pervar=10,
            burn_in=50,
            n_cores_MCMC=2,
            CornerPlots=True,  # TriangleFile='cornerplot.png',
            ChainPlots=True,
            PrintOptimStatus=True,
            parnames=[],
            mcmc_results=[],
            mcmc_bestparams=[],
            domain=[]):

        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            print("setting ", a_attribute, " to ", initlocals[a_attribute])
            setattr(self, a_attribute, initlocals[a_attribute])


    def MCMC(self, ZSetup, ASED, ZMerit):
        [names, mcmc_results] = exec_emcee(self, ZSetup, ASED, ZMerit)
        return [names, mcmc_results]

