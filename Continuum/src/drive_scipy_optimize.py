import sys
import os
import re
import numpy as np
import matplotlib
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt
import scipy.optimize as op

from pprint import pprint

from astropy import constants as const

from .Likelihood import lnlike
from .SummarySEDs import plotSEDs

c_MKS = const.c.value  # m/s
k_B = const.k_B.value



def run_scipy_optimize_minimize(x_free, names, bnds, ZSetup, ZData, ASED,
                                ZMerit, OptimM):
    #print("starting op.minimize")
    start_time = time()
    ftol = 1E-10  # 1e-10 too small leads to abnormal termination
    eps = []

    ndim = len(x_free)
    for j in range(ndim):
        lowerlimit = bnds[j][0]
        upperlimit = bnds[j][1]
        if (x_free[j] < lowerlimit):
            print("param", names[j], " is out of range - lowlimit ")
            x_free[j] = lowerlimit
        if (x_free[j] > upperlimit):
            print("param", names[j], " is out of range - uplimit ")
            x_free[j] = upperlimit

    for ibnd, abnd in enumerate(bnds):
        #if ((x_free[ibnd] < abnd[0]) or (x_free[ibnd] > abnd[1])):
        #    print("param ", names[ibnd], " is out of range")
        #    sys.exit()
        eps.append((abnd[1] - abnd[0]) / 100.)
    eps = np.array(eps)
    nll = lambda *args: -lnlike(*args)
    x_free_init = x_free.copy()
    #print("run_scipy_optimize_minimize",x_free_init)
    #print("ASED.nus",ASED.nus)
    ASED.calcul(ForcePrep=True)
    #ASED.calcul()

    ASED4alphas = None
    if ZData.nus_alphas is not None:
        ASED4alphas = MSED(ZSetup)
        ASED4alphas.copy(ASED)
        if ASED4alphas.GoInterp:
            ASED4alphas.gridfiletag = '_4alphas'
        ASED4alphas.nus = ZData.nus_alphas
        #print("in scipy optimize calcul 4alphas")
        ASED4alphas.calcul(ForcePrep=True)
        #ASED4alphas.calcul()

    init_lnlike = lnlike(x_free_init, names, ZSetup, ZData, ASED, ASED4alphas,
                         ZMerit)

    if OptimM.Report:
        print("scipy.op init_chi2 %e" % (-2. * init_lnlike))
    options = {'eps': eps}

    if OptimM.CGmaxiter:
        options['maxiter'] = OptimM.CGmaxiter

    result = op.minimize(
        nll,
        x_free,
        tol=ftol,
        bounds=bnds,
        method='Powell',  #  'Powell' Nelder-Mead
        options=options,
        args=(names, ZSetup, ZData, ASED, ASED4alphas, ZMerit))
    result_ml = result["x"]
    if OptimM.Report:
        print("result", result)
        print("Optim done in (elapsed time):", time() - start_time)
        print("Delta params", result_ml - x_free_init)

    init_lnlike2 = lnlike(x_free_init, names, ZSetup, ZData, ASED, ASED4alphas,
                          ZMerit)
    if OptimM.Report:
        print("init chi2 xcheck   %e" % (-2. * init_lnlike2))
    best_lnlike = lnlike(result_ml, names, ZSetup, ZData, ASED, ASED4alphas,
                         ZMerit)

    if OptimM.Report:
        print("init chi2 xcheck   %e" % (-2. * init_lnlike2))
    retvals = ZMerit.calcul(ZSetup, ZData, ASED, ASED4alphas=ASED4alphas)

    if OptimM.Report:
        print("Powell best_chi2 %e" % (-2. * best_lnlike))

    if OptimM.Report:
        print("computing errors with Hessian")

    tmp_i = np.zeros(len(result_ml))
    errors_ml = np.zeros(len(result_ml))
    #for i in list(range(len(result_ml))):
    #    tmp_i[i] = 1.0
    #    uncertainty_i = np.sqrt(result.hess_inv(tmp_i)[i])
    #    errors_ml[i] = uncertainty_i
    #    # print(('{0:12.4e} +- {1:.1e}'.format(result.x[i], uncertainty_i)))
    return (result_ml, errors_ml, retvals)


def exec_ConjGrad(OptimM, ZSetup, ZData, ASED, ZMerit):

    maxiter = OptimM.CGmaxiter
    ASED.nus = ZData.nus

    names = list(map((lambda x: x[0]), OptimM.domain))
    sample_params = list(map((lambda x: x[1]), OptimM.domain))
    bnds = list(map((lambda x: x[2]), OptimM.domain))
    nvar = len(list(names))

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

        for j in range(nvar):
            lowerlimit = bnds[j][0]
            upperlimit = bnds[j][1]
            if (sample_params[j] < lowerlimit):
                sample_params[j] = lowerlimit
            if (sample_params[j] > upperlimit):
                sample_params[j] = upperlimit
        for j in range(nvar):
            lowerlimit = bnds[j][0]
            upperlimit = bnds[j][1]
            bnds[j][0] = sample_params[j] - (sample_params[j] -
                                             lowerlimit) / 10.
            bnds[j][1] = sample_params[j] + (upperlimit -
                                             sample_params[j]) / 10.
            lowerlimit = bnds[j][0]
            upperlimit = bnds[j][1]
            if OptimM.Report:
                print("inherit- narrow bounds:")
                print(names[j], sample_params[j], lowerlimit, upperlimit)

    if OptimM.Report:
        print("nvar = ", nvar)

    x_free = np.array(sample_params)

    (result_ml, errors_ml,
     retvals) = run_scipy_optimize_minimize(x_free, names, bnds, ZSetup, ZData,
                                            ASED, ZMerit, OptimM)

    if OptimM.Report:
        np.save(ZSetup.outputdir + 'result_ml.dat', result_ml)
        np.save(ZSetup.outputdir + 'result_ml_errors.dat', errors_ml)

    if OptimM.SummaryPlots:

        plotSEDs(
            nvar,
            names,
            ASED,
            ZData,
            ZSetup,
            CGbestparams=result_ml,
            mcmc_results=OptimM.mcmc_results,  # None,
            mcmc_bestparams=OptimM.mcmc_bestparams,  #None
            mcmc_results_0=OptimM.mcmc_results_0,
            chains=None,
            label4SED=OptimM.label4SED,
            trueparams=OptimM.trueparams,
            xscale=OptimM.summarySED_xaxisscale,
            yscale=OptimM.summarySED_yaxisscale,
            nchains_4plots=False,
            filename='fig_bestfit_Powell' + OptimM.filename_tag + '.png',
            DoubleArrow=False)

    achi2 = retvals[0]
    modelInus = retvals[1]
    if ZData.nus_alphas is not None:
        modelalphas = retvals[2]
    else:
        modelalphas = []

    if OptimM.Report:
        ZSetup.filetag = 'powell_bestL_'

    achi2 = retvals[0]
    modelInus = retvals[1]
    if ZData.nus_alphas is not None:
        modelalphas = retvals[2]
    else:
        modelalphas = []

    #print("retvals",retvals)
    if OptimM.Report:
        print("best powell chi2 = %e" % (achi2))

    #return names, result_ml
    return [names, result_ml, modelInus, modelalphas, achi2]



