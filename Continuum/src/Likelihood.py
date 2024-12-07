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
         with_specindexdata=False,
         ASED4alphas=False,
         Regul=False):

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
        LbdaMassRegul = 1E1
        #LbdaMassRegul = 0.
        if ASED.Sigma_g > ASED.Sigma_g_0:
            SMassRegul = ((ASED.Sigma_g - ASED.Sigma_g_0) / ASED.Sigma_g_0)**2
            regulterm += LbdaMassRegul * SMassRegul
            #print("REgularizing Sigma ", chi2, SMassRegul)

        amax_c = 0.1 # cm
        LbdaamaxRegul=0.
        if ASED.amax > amax_c:
            SamaxRegul = ((ASED.amax - amax_c) / amax_c)**2
            regulterm += LbdaamaxRegul * SamaxRegul
            #print("REgularizing Sigma ", chi2, SMassRegul)
            
        MaxOptiDepth = 2.
        LbdaOpacRegul = 1E2
        inulowest = np.argmin(ZData.nus)
        lowesttau = ASED.tau[inulowest]
        if (lowesttau > MaxOptiDepth):
            SOpacRegul = (lowesttau - MaxOptiDepth)**2
            regulterm += LbdaOpacRegul * SOpacRegul
            #print("REgularizing Opac ", chi2, SOpacRegul)
        chi2 += regulterm
        retvals[0] = chi2

    return retvals
