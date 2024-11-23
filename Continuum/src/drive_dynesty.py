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
#from multiprocessing import Pool

import scipy.optimize as op
import dynesty
from dynesty.pool import Pool
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from dynesty.utils import resample_equal

from pprint import pprint

from .SEDOptim import lnlike
from .SEDOptim import initoptim
from .SEDOptim import lnlike_4pool


def prior_transform(unif_free, OptimM):
    """Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameter of interest `x ~ Unif[-10., 10.)`."""
    x_free = np.zeros_like(unif_free)
    bnds = list(map((lambda x: x[2]), OptimM.domain))
    for iparam in list(range(len(unif_free))):
        x_free[iparam] = bnds[iparam][0] + (bnds[iparam][1] - bnds[iparam][0]) * unif_free[iparam]

    return x_free


def exec_dynesty(OptimM, ZSetup, ZData, ASED, ZMerit, Debug=False):
    workdir = ZSetup.outputdir

    names, sample_params, bnds, nvar, x_free_init, init_lnlike = initoptim(
        OptimM, ZSetup, ZData, AModel, ZMerit, Debug=Debug
    )

    xcheck_init_profiles = True
    if xcheck_init_profiles:
        ZSetup.filetag = 'init_'
        OptimM.plotprofiles(AModel, ZSetup, basename_file_profiles=ZSetup.filetag+'fig_profiles.pdf')
        OptimM.skyfig(ZData, ZSetup, AModel, WithSurface=True)
        ZSetup.filetag = None

    Nit = OptimM.maxiter
    ngpus = ZSetup.ngpus

    if OptimM.nlive_pervar is not None:
        nlive = OptimM.nlive_pervar * nvar
        nlive = ngpus * int(nlive / ngpus)
        nlive_perGPU = int(nlive / ngpus)
        if nlive_perGPU < (2 * nvar):
            print("WARNING: nlive_perGPU < (2*nvar)")
            nlive_perGPU = int(2 * nvar + 1)
            nlive = int(ngpus * nlive_perGPU)
            print("reset nlive to ", nlive, nlive_perGPU)
    else:
        nlive_perGPU = None

    print("nlive", nlive)

    ndim = nvar

    print("in exec_dynesty with RunSampler=", OptimM.RunSampler)
    start_time = time.time()
    t_i = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    if OptimM.RunSampler:
        nthreads = ngpus
        lnlikeargs = [names, ZSetup, ZData, AModel, ZMerit]  # , iGPU]
        lnlikekwargs = {'Debug': Debug, 'UseHandle': True}
        ptform_args = [OptimM]

        print("Setting up Dynesty  sampler with n_live = ", nlive)
        with ZSetup.ctx.Pool(processes=ngpus) as pool:
            d_args = {}
            d_args['bound'] = 'multi'
            d_args['sample'] = 'rwalk'
            d_args['nlive'] = nlive
            d_args['logl_args'] = lnlikeargs
            d_args['logl_kwargs'] = lnlikekwargs
            d_args['ptform_args'] = ptform_args

            sampler = dynesty.DynamicNestedSampler(
                lnlike_4pool,
                prior_transform,
                ndim,
                queue_size=ngpus,
                pool=pool,
                enlarge=1,
                **d_args
            )
            sampler.run_nested()

        dresult = sampler.results
        print("dresult.keys()", dresult.keys())
        points = dresult.samples  # dresult['samples']
        log_w = dresult['logwt']
        log_z = dresult['logz']
        log_l = dresult['logl']
        np.save(workdir + 'dresult_points', points)
        np.save(workdir + 'dresult_log_w', log_w)
        np.save(workdir + 'dresult_log_z', log_z)
        np.save(workdir + 'dresult_log_l', log_l)

        points_equalw = dresult.samples_equal
        np.save(workdir + 'dresult_points_equalw', points_equalw)

        t_f = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        Delta_t = (time.time() - start_time) / 3600.
        print("Execution done in (elapsed time):" + str(Delta_t) + " h")
        print("t_i = " + str(t_i))
        print("t_f = " + str(t_f))

    else:
        points = np.load(workdir + 'dresult_points.npy', allow_pickle=True)
        log_w = np.load(workdir + 'dresult_log_w.npy', allow_pickle=True)
        log_z = np.load(workdir + 'dresult_log_z.npy', allow_pickle=True)
        log_l = np.load(workdir + 'dresult_log_l.npy', allow_pickle=True)
        points_equalw = np.load(workdir + 'dresult_points_equalw.npy', allow_pickle=True)
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

    if OptimM.CornerPlots:
        print("corner")
        # Plot the 2-D marginalized posteriors.
        cfig, caxes = dyplot.cornerplot(dresult)
        cfig.savefig(workdir + 'corner.png')

    print("get mean and covariances")

    ibestparams = np.argmax(log_l)
    bestparams = points[ibestparams, :]

    # Compute 10%-90% quantiles.
    #quantiles = [
    #    dyfunc.quantile(samps, [2.5, 50, 97.5], weights=weights)
    #    for samps in samples.T
    #]
    quantiles = [dyfunc.quantile(samps, [0.25, 0.50, 0.975], weights=log_w) for samps in points.T]
    print("quantiles", quantiles)

    # print("points_equalw.shape", points_equalw.shape)
    # quantiles_eqw = [
    #     dyfunc.quantile(samps, [0.25, 0.50, 0.975])
    #     for samps in points_equalw.T
    # ]
    # print("quantiles_eqw", quantiles_eqw)

    weights = np.exp(log_w - log_z[-1])
    posterior_samples = resample_equal(points, weights)
    
    mcmc_results = list(
        map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(posterior_samples, [2.5, 50, 97.5], axis=0))
        )
    )

    print("sampler results", mcmc_results)

    # tips from Rafael B.:
    #weights = np.exp(dresult['logwt'] - dresult['logz'][-1])
    #posterior_samples = resample_equal(dresult.samples, weights)
    # ###
    #posterior_samples = points_equalw
    #posterior_samples = points
    OptimM.mcmc_samples = posterior_samples

    mcmc_results_0 = np.zeros(nvar)
    with open(ZSetup.outputdir + "bestfitparams.txt", "w") as f:
        print("%20s %12s %12s %12s %12s\n" % ('#name', 'median', 'uperror', 'loerror', 'maxlnlike'))
        f.write(
            "%20s %12s %12s %12s %12s\n" % ('#name', 'median', 'uperror', 'loerror', 'maxlnlike')
        )
        mcmc_results_uperrs = np.zeros(nvar)
        mcmc_results_doerrs = np.zeros(nvar)
        for iparam in list(range(nvar)):
            print(names[iparam], mcmc_results[iparam], bestparams[iparam])
            print(
                "%20s %12.4e %12.4e %12.4e %12.4e" % (
                    names[iparam], mcmc_results[iparam][0], mcmc_results[iparam][1],
                    mcmc_results[iparam][2], bestparams[iparam]
                ),
                file=f
            )
            mcmc_results_0[iparam] = mcmc_results[iparam][0]
            mcmc_results_uperrs[iparam] = mcmc_results[iparam][1]
            mcmc_results_doerrs[iparam] = mcmc_results[iparam][2]

            #cornerquantiles=corner.quantile(points[:,iparam], [0.25, 0.50, 0.975], weights=np.exp(log_w))
            #uperr = cornerquantiles[2]-cornerquantiles[1]
            #loerr = cornerquantiles[1]-cornerquantiles[0]
            #print("corner ",cornerquantiles[0],uperr, loerr)
            #print("equalw ",mcmc_results_equalw[iparam])
    f.close()

    print("mcmc_results_uperrs", mcmc_results_uperrs)
    getcoeferrs(mcmc_results_uperrs, mcmc_results_doerrs, names, AModel, ZSetup, iGPU=0)

    OptimM.mcmc_results = mcmc_results
    OptimM.mcmc_bestparams = bestparams
    OptimM.mcmc_mean = mcmc_results_0

    #uncomment to extract profile statistics:
    #OptimM.getstats(AModel, ZSetup, iGPU=0, nsamples=100)

    print("Finished sampler for  workdir", workdir)

    #print("cross check  lnlike for init conditions still the same")
    #print("x_free_init", x_free_init)
    #init_lnlike2 = lnlike(x_free_init,
    #                      names,
    #                      ZSetup,
    #                      ZData,
    #                      AModel,
    #                      ZMerit,
    #                      iGPU=0)
    #print("chi2 init = %e " % (-2 * init_lnlike2))
    print("running final lnlike to set model to mean values")
    ZSetup.filetag = 'mcmc_mean_'
    final_lnlike = lnlike(mcmc_results_0, names, ZSetup, ZData, AModel, ZMerit, Debug=False, iGPU=0)
    print("chi2 mean = %e " % (-2 * final_lnlike))
    AModel.punchprofiles(ZSetup, iGPU=0)
    OptimM.plotprofiles(AModel, ZSetup, basename_file_profiles=ZSetup.filetag+'fig_profiles.pdf')
    OptimM.skyfig(ZData, ZSetup, AModel, WithSurface=True)

    print("running final lnlike to set model to best values")
    ZSetup.filetag = 'mcmc_best_'
    final_lnlike = lnlike(bestparams, names, ZSetup, ZData, AModel, ZMerit, Debug=False, iGPU=0)
    print("chi2 best params = %e " % (-2 * final_lnlike))
    AModel.punchprofiles(ZSetup, iGPU=0)
    OptimM.plotprofiles(AModel, ZSetup, basename_file_profiles=ZSetup.filetag+'fig_profiles.pdf')
    OptimM.skyfig(ZData, ZSetup, AModel, WithSurface=True)

    return [names, mcmc_results]
