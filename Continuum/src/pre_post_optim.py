import sys
import os
import re
import numpy as np
from time import time
from copy import deepcopy

from pprint import pprint
from astropy import constants as const
#from astropy.constants import astropyconst20 as const

from .Likelihood import lnlike

c_MKS = const.c.value  # m/s
k_B = const.k_B.value

def Tbrightness(I_nu, nu):
    # input I_nu in Jy/srx
    # input nu in Hz
    if (I_nu < 0.):
        Tb = 1.
    else:
        h_P = const.h.value
        k_B = const.k_B.value
        c_light = const.c.value
        Tb = h_P * nu / (k_B * np.log(1. + (2. * h_P * nu**3 /
                                            (c_light**2 * I_nu * 1E-26))))

    return Tb



def initoptim(OptimM, ZSetup, ZData, ASED, ZMerit):
    """ 
    initialize optimization
    OptimM.Inherit_Init: inherit initial conditions from ASED rather than from OptimM.domain
    """
    ASED.nus = ZData.nus

    names = list(map((lambda x: x[0]), OptimM.domain))
    sample_params = list(map((lambda x: x[1]), OptimM.domain))
    bnds = list(map((lambda x: x[2]), OptimM.domain))
    nvar = len(list(names))

    #pprint(OptimM.domain)

    #initial conditions
    Tdust_init = 30.
    Sigma_g_init = 1.
    inus_sorted = np.argsort(ZData.nus)
    for iname, aname in enumerate(names):
        if 'Tdust' in aname:
            inus_sorted_desc = np.flip(inus_sorted)
            found = False
            for inu_highest in inus_sorted_desc:
                nu_highest = ZData.nus[inu_highest]
                Inu_highest = ZData.Inus[inu_highest]
                if Inu_highest > 0.:
                    Tdust_init = Tbrightness(Inu_highest, nu_highest)
                    if OptimM.PhysicalInit:
                        dolog = False
                        m = re.search('log\((.*)\)', aname)
                        if m:
                            dolog = True
                        if dolog:
                            sample_params[iname] = np.log10(Tdust_init)
                        else:
                            sample_params[iname] = Tdust_init
                    found=True
                    break

            if not found:
                sys.exit("pre_post_optim, Tdust_init: only noise here")

            ASED.Tdust_0 = Tdust_init
            if OptimM.Report:
                print("Tdust_init", Tdust_init)
                
    for iname, aname in enumerate(names):
        if 'Sigma_g' in aname:
            beta = 1
            inus_sorted = np.argsort(ZData.nus)
            found = False
            for inu_lowest in inus_sorted:
                if ZData.Inus[inu_lowest] > 0:
                    nu_lowest = ZData.nus[inu_lowest]
                    Inu_lowest = ZData.Inus[inu_lowest]
                    found = True
                    break
            if not found:
                sys.exit("pre_post_optim, Sigma_g_0: only noise here")
                    
            kappa_nu = 0.02 * (nu_lowest / 230609583076.92312)**beta
            # cm2 / g , Beckwith
            #  (0.02 cm^2/g / kappa_1.3mm)

            # RJ = (2. * k_B * nu_lowest**2 / c_light**2) Tb
            # I = RJ * tau
            tau0 = Inu_lowest * 1E-26 / (
                (2. * k_B * nu_lowest**2 / c_MKS**2) * Tdust_init)
            Sigma_g_init = tau0 / kappa_nu
            ASED.Sigma_g_0 = Sigma_g_init
            if OptimM.PhysicalInit and (Inu_lowest > 0):
                m = re.search('log\((.*)\)', aname)
                dolog = False
                if m:
                    dolog = True
                if dolog:
                    sample_params[iname] = np.log10(Sigma_g_init)
                else:
                    sample_params[iname] = Sigma_g_init
            if OptimM.Report:
                print("Sigma_g_init", Sigma_g_init)

    if OptimM.Inherit_Init:
        for iname, aname in enumerate(names):
            aname_4value = aname
            m = re.search('log\((.*)\)', aname)
            dolog = False
            if m:
                dolog = True
                aname_4value = m.group(1)
            value = getattr(ASED, aname_4value)
            if dolog:
                value = np.log10(value)
            sample_params[iname] = value
            if OptimM.Report:
                print("Inherit_Init: ", aname, sample_params[iname])

    if OptimM.Report:
        print("nvar = ", nvar)

    x_free = np.array(sample_params)
    x_free_init = x_free.copy()

    if OptimM.Report:
        print("check initial chi2 value")

    ASED4alphas = None
    if ZMerit.with_specindexdata:
        #ASED4alphas = AModelSED.MSED(ZSetup)
        #ASED4alphas.copy(ASED) DEV
        ASED4alphas=deepcopy(ASED)
        if ASED4alphas.GoInterp:
            ASED4alphas.gridfiletag = '_4alphas'
        ASED4alphas.nus = ZData.nus_alphas
        ASED4alphas.calcul(ForcePrep=True)

    init_lnlike = lnlike(x_free, names, ZSetup, ZData, ASED, ASED4alphas,
                         ZMerit)
    if OptimM.Report:
        print("init lnlike  = %e " % (init_lnlike))
        print("chi2 = %e " % (-2 * init_lnlike))

    return nvar, sample_params, names, bnds, ASED4alphas, x_free_init

