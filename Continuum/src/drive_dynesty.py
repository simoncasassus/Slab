import sys
import os
import os.path
import numpy as np
import math
from astropy.io import fits
import time
import astropy.constants as const
import re
import random
from multiprocessing import Pool

#from multiprocessing import get_context

import scipy.optimize as op
import dynesty
from dynesty.pool import Pool
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from dynesty.utils import resample_equal

from pprint import pprint

from .Likelihood import lnlike
from .pre_post_optim import initoptim
from .SummarySEDs import plotSEDs


def prior_transform(unif_free, OptimM):
    """Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameter of interest `x ~ Unif[-10., 10.)`."""
    x_free = np.zeros_like(unif_free)
    bnds = list(map((lambda x: x[2]), OptimM.domain))
    for iparam in list(range(len(unif_free))):
        x_free[iparam] = bnds[iparam][0] + (
            bnds[iparam][1] - bnds[iparam][0]) * unif_free[iparam]

    return x_free


def exec_dynesty(OptimM, ZSetup, ZData, ASED, ZMerit, Debug=False):
    workdir = ZSetup.outputdir

    nvar, sample_params, names, bnds, ASED4alphas, x_free_init = initoptim(
        OptimM, ZSetup, ZData, ASED, ZMerit)

    Nit = OptimM.maxiter

    nlive = OptimM.nlive_pervar * nvar

    ndim = nvar

    start_time = time.time()
    t_i = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    if OptimM.RunSampler:

        nthreads = OptimM.n_cores_sampler
        lnlikeargs = [names, ZSetup, ZData, ASED, ASED4alphas, ZMerit]

        ptform_args = [OptimM]

        if OptimM.Report:
            print("Setting up Dynesty  sampler with n_live = ", nlive,
                  " and nthreads", nthreads, " <<")

        d_args = {}
        d_args['bound'] = 'multi'
        d_args['sample'] = 'rwalk'
        d_args['nlive'] = nlive
        d_args['logl_args'] = lnlikeargs
        #d_args['logl_kwargs'] = lnlikekwargs
        d_args['ptform_args'] = ptform_args
        #nthreads = 1
        if nthreads > 1:
            #ctx = get_context('spawn')
            with Pool(nthreads, lnlike, prior_transform) as pool:
                #with Pool(nthreads) as pool:
                #sampler = NestedSampler(pool.loglikehood, pool.prior_transform,
                #            ndim, pool = pool)
                sampler = dynesty.DynamicNestedSampler(pool.loglikelihood,
                                                       pool.prior_transform,
                                                       ndim,
                                                       queue_size=nthreads,
                                                       pool=pool,
                                                       enlarge=1,
                                                       **d_args)

        else:
            sampler = dynesty.DynamicNestedSampler(
                lnlike,
                prior_transform,
                ndim,
                queue_size=nthreads,
                enlarge=1,
                **d_args,
            )

        sampler.run_nested(print_progress=OptimM.Report)
        dresult = sampler.results
        points = dresult.samples  # dresult['samples']
        log_w = dresult['logwt']
        log_z = dresult['logz']
        log_l = dresult['logl']
        points_equalw = dresult.samples_equal
        if OptimM.Report:
            print("dresult.keys()", dresult.keys())
            np.save(workdir + 'dresult_points', points)
            np.save(workdir + 'dresult_log_w', log_w)
            np.save(workdir + 'dresult_log_z', log_z)
            np.save(workdir + 'dresult_log_l', log_l)
            np.save(workdir + 'dresult_points_equalw', points_equalw)

            t_f = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            Delta_t = (time.time() - start_time) / 3600.
            print("Execution done in (elapsed time):" + str(Delta_t) + " h")
            print("t_i = " + str(t_i))
            print("t_f = " + str(t_f))
        if OptimM.CornerPlots:
            print("corner")
            # Plot the 2-D marginalized posteriors.
            cfig, caxes = dyplot.cornerplot(dresult)
            cfig.savefig(workdir + 'corner' + OptimM.filename_tag + '.png')

    else:
        points = np.load(workdir + 'dresult_points.npy', allow_pickle=True)
        log_w = np.load(workdir + 'dresult_log_w.npy', allow_pickle=True)
        log_z = np.load(workdir + 'dresult_log_z.npy', allow_pickle=True)
        log_l = np.load(workdir + 'dresult_log_l.npy', allow_pickle=True)
        points_equalw = np.load(workdir + 'dresult_points_equalw.npy',
                                allow_pickle=True)
        #log_w_equalw = np.load(workdir + 'dresult_log_w_equalw.npy',
        #                       allow_pickle=True)
        #log_l_equalw = np.load(workdir + 'dresult_log_l_equalw.npy',
        #                       allow_pickle=True)

    #print("summary plot")
    # Plot a summary of the run.
    #rfig, raxes = dyplot.runplot(dresult)
    #rfig.savefig(workdir + 'summary.png')

    #print("traces")
    ## Plot traces and 1-D marginalized posteriors.
    #tfig, taxes = dyplot.traceplot(dresult)
    #tfig.savefig(workdir + 'traces.png')


    ibestparams = np.argmax(log_l)
    bestparams = points[ibestparams, :]

    # Compute 10%-90% quantiles.
    #quantiles = [
    #    dyfunc.quantile(samps, [2.5, 50, 97.5], weights=weights)
    #    for samps in samples.T
    #]
    quantiles = [
        dyfunc.quantile(samps, [0.25, 0.50, 0.975], weights=log_w)
        for samps in points.T
    ]
    if OptimM.Report:
        print("quantiles", quantiles)

    # print("points_equalw.shape", points_equalw.shape)
    # quantiles_eqw = [
    #     dyfunc.quantile(samps, [0.25, 0.50, 0.975])
    #     for samps in points_equalw.T
    # ]
    # print("quantiles_eqw", quantiles_eqw)

    weights = np.exp(log_w - log_z[-1])
    posterior_samples = resample_equal(points, weights)

    #mcmc_results = list(
    #    map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
    #        zip(*np.percentile(posterior_samples, [2.5, 50, 97.5], axis=0))))
    mcmc_results = list(
        map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(posterior_samples, [16, 50, 84], axis=0))))

    if OptimM.Report:
        print("sampler results", mcmc_results)

    # tips from Rafael B.:
    #weights = np.exp(dresult['logwt'] - dresult['logz'][-1])
    #posterior_samples = resample_equal(dresult.samples, weights)
    # ###
    #posterior_samples = points_equalw
    #posterior_samples = points
    OptimM.mcmc_samples = posterior_samples

    mcmc_results_0 = np.zeros(nvar)
    mcmc_results_uperrs = np.zeros(nvar)
    mcmc_results_doerrs = np.zeros(nvar)
    for iparam in list(range(nvar)):
        mcmc_results_0[iparam] = mcmc_results[iparam][0]
        mcmc_results_uperrs[iparam] = mcmc_results[iparam][1]
        mcmc_results_doerrs[iparam] = mcmc_results[iparam][2]
    if OptimM.Report:
        with open(ZSetup.outputdir + "bestfitparams.txt", "w") as f:
            print("%20s %12s %12s %12s %12s\n" %
                  ('#name', 'median', 'uperror', 'loerror', 'maxlnlike'))
            f.write("%20s %12s %12s %12s %12s\n" %
                    ('#name', 'median', 'uperror', 'loerror', 'maxlnlike'))
            for iparam in list(range(nvar)):
                print(names[iparam], mcmc_results[iparam], bestparams[iparam])
                print("%20s %12.4e %12.4e %12.4e %12.4e" %
                      (names[iparam], mcmc_results[iparam][0],
                       mcmc_results[iparam][1], mcmc_results[iparam][2],
                       bestparams[iparam]),
                      file=f)

                #cornerquantiles=corner.quantile(points[:,iparam], [0.25, 0.50, 0.975], weights=np.exp(log_w))
                #uperr = cornerquantiles[2]-cornerquantiles[1]
                #loerr = cornerquantiles[1]-cornerquantiles[0]
                #print("corner ",cornerquantiles[0],uperr, loerr)
                #print("equalw ",mcmc_results_equalw[iparam])
        f.close()

    if OptimM.Report:
        print("mcmc_results_uperrs", mcmc_results_uperrs)

    OptimM.mcmc_results = mcmc_results
    OptimM.mcmc_bestparams = bestparams
    OptimM.mcmc_results_0 = mcmc_results_0

    if OptimM.Report:
        print("cross check initial chi2 value")
        init_lnlike = lnlike(x_free_init, names, ZSetup, ZData, ASED,
                             ASED4alphas, ZMerit)
        print("chi2 = %e " % (-2 * init_lnlike))

    if OptimM.Report:
        print("running final lnlike to set model to median values")
        final_lnlike = lnlike(mcmc_results_0, names, ZSetup, ZData, ASED,
                              ASED4alphas, ZMerit)
        print("chi2 = %e " % (-2 * final_lnlike))

    if OptimM.Report:
        print("running final lnlike to set model to best L  values")
        ZSetup.filetag = 'mcmc_bestL_'
    final_lnlike = lnlike(bestparams, names, ZSetup, ZData, ASED, ASED4alphas,
                          ZMerit)
    retvals = ZMerit.calcul(ZSetup, ZData, ASED, ASED4alphas=ASED4alphas)
    achi2 = retvals[0]
    modelInus = retvals[1]
    if ZData.nus_alphas is not None:
        modelalphas = retvals[2]
    else:
        modelalphas = []

    #print("retvals",retvals)
    if OptimM.Report:
        print("chi2 = %e" % (achi2))

    ######################################################################
    # plot results

    if OptimM.SummaryPlots:
        nchains_4plots = Nit
        if (nchains_4plots > 300):
            nchains_4plots = 300

        plotSEDs(nvar,
                 names,
                 ASED,
                 ZData,
                 ZSetup,
                 CGbestparams=None,
                 mcmc_results=mcmc_results,
                 mcmc_bestparams=bestparams,
                 mcmc_results_0=mcmc_results_0,
                 trueparams=OptimM.trueparams,
                 xscale=OptimM.summarySED_xaxisscale,
                 yscale=OptimM.summarySED_yaxisscale,
                 WithSEDchains=OptimM.summaryWithSEDchains,
                 nchains_4plots=nchains_4plots,
                 label4SED=OptimM.label4SED,
                 filename='fig_bestfit' + OptimM.filename_tag + '.png',
                 DoubleArrow=False)

    return [names, mcmc_results, bestparams, modelInus, modelalphas, achi2]
