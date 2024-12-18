import sys
import os
import re
import numpy as np
import matplotlib
from time import time
from copy import deepcopy
from scipy.interpolate import interp1d
import scipy.optimize as op
import cmath as cma

from pprint import pprint

from astropy import constants as const
#from astropy.constants import astropyconst20 as const

c_MKS = const.c.value  # m/s
k_B = const.k_B.value


def assignfreeparams(parnames, values, ASED):

    for iparam, aparam in enumerate(parnames):
        aname = aparam
        avalue = values[iparam]
        m = re.search('log\((.*)\)', aname)
        if m:
            aname = m.group(1)
            avalue = 10**(avalue)
        #print("ASED assign:",aname,"<- ",avalue)
        setattr(ASED, aname, avalue)


def lnlike(x_free, parnames, ZSetup, ZData, ASED, ASED4alphas, ZMerit):

    assignfreeparams(parnames, x_free, ASED)

    chi2 = ZMerit.calcul(ZSetup, ZData, ASED, ASED4alphas=ASED4alphas)[0]

    # chi2 = chi2.item()
    # if OptimM.PrintChi2s:
    # print("lnlike chi2 %e " % (chi2))

    return -0.5 * chi2


def logL(ZData,
         ASED,
         LbdaSigma_gRegul = 1.,
         MaxOptDepth = 3.,
         LbdaOptDepthRegul = 10.,
         LbdaamaxRegul = 1.,
         LbdaTdustRegul=0.,
         with_specindexdata=False,
         ASED4alphas=False,
         Regul=False):
    """
    Regularization:
    LbdaSigma_gRegul = 1.  -> Weight for Sigma_g regul
    MaxOptDepth = 3.  -> maximum optical depth at lowest freq
    LbdaOptDepthRegul = 1E3 -> weight for optical depth regul 
    LbdaamaxRegul = 1. -> weight for amax regular
    """

    if (np.sum(ZData.sInus) == 0):
        chi2 = np.array([0.])
    else:
        chi2 = np.sum((ZData.Inus - ASED.Inus)**2 / ZData.sInus**2)

    retvals = [chi2, ASED.Inus]
    # print("chi2  = ", chi2, " dofs ", len(ZData.Inus), ASED.Inus, ASED.Tdust, ASED.Sigma_g, ASED.amax, ASED.q_dustexpo, ZData.sInus)

    if with_specindexdata:
        npairs = int(len(ASED4alphas.nus) / 2)
        alphas = np.zeros(npairs)
        nu1s = np.zeros(npairs)
        nu2s = np.zeros(npairs)
        for ipair in range(npairs):
            alphas[ipair] = np.log(ASED4alphas.Inus[2 * ipair + 1] /
                                   ASED4alphas.Inus[2 * ipair]) / np.log(
                                       ASED4alphas.nus[2 * ipair + 1] /
                                       ASED4alphas.nus[2 * ipair])

        chi2_alphas = np.sum((ZData.alphas - alphas)**2 / ZData.salphas**2)
        # print("chi2_alphas  = ", chi2_alphas, " dofs ", len(ZData.alphas))
        chi2 += chi2_alphas
        retvals.append(alphas)
        retvals[0] = chi2
    # print("chi2  = ", chi2)

    if Regul:
        regulterm = 0.

        bracketTdust = True
        if bracketTdust:
            scaleuplimit = 1.
            if ASED.Tdust > scaleuplimit * ASED.Tdust_high:
                STdustRegul = ((ASED.Tdust - scaleuplimit * ASED.Tdust_high) / (scaleuplimit * ASED.Tdust_high))**2
                regulterm += LbdaTdustRegul * STdustRegul

            if ASED.Tdust < ASED.Tdust_low:
                STdustRegul = ((ASED.Tdust - ASED.Tdust_low) / ASED.Tdust_low)**2
                regulterm += LbdaTdustRegul * STdustRegul
        else:
            if ASED.Tdust > ASED.Tdust_0:
                STdustRegul = ((ASED.Tdust - ASED.Tdust_0) / ASED.Tdust_0)**2
                regulterm += LbdaTdustRegul * STdustRegul

        
        bracketSigma_g = True
        if bracketSigma_g:
            if ASED.Sigma_g > ASED.Sigma_g_high:
                SSigma_gRegul = ((ASED.Sigma_g - ASED.Sigma_g_high) / ASED.Sigma_g_high)**2
                regulterm += LbdaSigma_gRegul * SSigma_gRegul

            if ASED.Sigma_g < ASED.Sigma_g_low:
                SSigma_gRegul = ((ASED.Sigma_g - ASED.Sigma_g_low) / ASED.Sigma_g_low)**2
                regulterm += LbdaSigma_gRegul * SSigma_gRegul

        else:
            if ASED.Sigma_g > ASED.Sigma_g_0:
                SMassRegul = ((ASED.Sigma_g - ASED.Sigma_g_0) / ASED.Sigma_g_0)**2
                regulterm += LbdaSigma_gRegul * SMassRegul
            #print("REgularizing Sigma ", chi2, SMassRegul)

        amax_c = 1.  # cm
        if ASED.amax > amax_c:
            SamaxRegul = ((ASED.amax - amax_c) / amax_c)**2
            regulterm += LbdaamaxRegul * SamaxRegul
            #print("REgularizing Sigma ", chi2, SMassRegul)

        inulowest = np.argmin(ZData.nus)
        lowesttau = ASED.tau[inulowest]
        if (lowesttau > MaxOptDepth):
            SOpacRegul = (lowesttau - MaxOptDepth)**2
            regulterm += LbdaOptDepthRegul * SOpacRegul
            #print("REgularizing Opac ", chi2, SOpacRegul)
        chi2 += regulterm
        retvals[0] = chi2

    return retvals
