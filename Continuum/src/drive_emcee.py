import sys
import os
import re
import numpy as np
import matplotlib
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt

from pprint import pprint
import emcee
import corner

from .SummarySEDs import plotSEDs

from multiprocessing import Pool

from astropy import constants as const

c_MKS = const.c.value  # m/s
k_B = const.k_B.value

#HOME = os.environ.get('HOME')
#include_path = HOME + '/gitcommon/'
#sys.path.append(include_path)

from .Likelihood import lnlike
from .pre_post_optim import initoptim
from .SummarySEDs import plotSEDs

def lnprior(theta, bnds):
    inside = 1
    #bnds = list(map((lambda x: x[2]), OptimM.domain))
    for iparam in list(range(len(theta))):
        if (bnds[iparam][0] < theta[iparam] < bnds[iparam][1]):
            inside *= 1
        else:
            inside *= 0
    if (inside):
        return 0.0
    else:
        return -np.inf


def lnprob(x_free, parnames, bnds, ZSetup, ZData, ASED, ASED4alphas, ZMerit):

    lp = lnprior(x_free, bnds)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(x_free, parnames, ZSetup, ZData, ASED, ASED4alphas,
                       ZMerit)


def exec_emcee(OptimM, ZSetup, ZData, ASED, ZMerit):

    workdir = ZSetup.outputdir

    nvar, sample_params, names, bnds, ASED4alphas, x_free_init = initoptim(
        OptimM, ZSetup, ZData, ASED, ZMerit)

    Nit = OptimM.MCMC_Nit
    nwalkers = OptimM.nwalkers_pervar * nvar
    n_cores = OptimM.n_cores_MCMC
    burn_in = OptimM.burn_in  # 100

    ranges = list(map((lambda x: x[2][1] - x[2][0]), OptimM.domain))
    allowed_ranges = np.array(ranges)

    ndim = nvar
    pos = []
    for i in list(range(nwalkers)):
        if (np.any(allowed_ranges < 0.)):
            sys.exit("wrong order of bounds in domains")
        awalkerinit = sample_params + (OptimM.mcmcinitball_relrange *
                                       np.random.randn(ndim) * allowed_ranges)
        #awalkerinit = sample_params + (np.random.randn(ndim) * allowed_ranges)

        for j in list(range(ndim)):
            lowerlimit = bnds[j][0]
            upperlimit = bnds[j][1]
            if (awalkerinit[j] < lowerlimit):
                awalkerinit[j] = lowerlimit
            if (awalkerinit[j] > upperlimit):
                awalkerinit[j] = upperlimit

        pos.append(awalkerinit)

    # os.environ["OMP_NUM_THREADS"] = "1"
    if OptimM.Report:
        print("in exec_emcee with RunMCMC=", OptimM.RunMCMC)
    if OptimM.RunMCMC:
        if OptimM.Report:
            print("now about to call run_mcmc with Nit", Nit, "and nmwalkers",
                  nwalkers, " and ncores", n_cores)

        start = time()
        lnprobargs = [names, bnds, ZSetup, ZData, ASED, ASED4alphas, ZMerit]

        if n_cores > 1:
            with Pool(n_cores) as pool:
                sampler = emcee.EnsembleSampler(nwalkers,
                                                ndim,
                                                lnprob,
                                                args=lnprobargs,
                                                pool=pool)
                sampler.run_mcmc(pos, Nit, progress=OptimM.MCMCProgress)
        else:

            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            lnprob,
                                            args=lnprobargs)
            sampler.run_mcmc(pos, Nit, progress=OptimM.MCMCProgress)

        end = time()
        emcee_time = end - start
        if OptimM.Report:
            print("EnsembleSampler took {0:.1f} seconds".format(emcee_time))

        samples = sampler.chain  # chain= array(nwalkers,nit,ndim)
        lnprobs = sampler.lnprobability

        if OptimM.Report:
            ######### save samples
            np.save(workdir + 'samples.dat', samples)
            np.save(workdir + 'lnprobs.dat', lnprobs)
            Delta_t = emcee_time / 60.
            print("Execution done in (elapsed time):" + str(Delta_t) + " mn")

            print(("mean acceptance fraction: {0:.3f} ".format(
                np.mean(sampler.acceptance_fraction))))
            f = open(workdir + 'acceptance.dat', 'w')
            f.write("Execution done in:" + str(Delta_t) + "h")
            f.write("Nit = " + str(Nit) + ' \n')
            f.write("nwalkers = " + str(nwalkers) + ' \n')
            f.write("ndim = " + str(ndim) + ' \n')
            f.write("mean acceptance fraction: {0:.3f}".format(
                np.mean(sampler.acceptance_fraction)) + ' \n')
            f.close()
    else:
        samples = np.load(workdir + 'samples.dat.npy')
        lnprobs = np.load(workdir + 'lnprobs.dat.npy')

    chains = np.zeros(((Nit - burn_in) * nwalkers, ndim))
    lnpchain = np.zeros(((Nit - burn_in) * nwalkers))

    chains[:, :] = samples[:, burn_in:, :].reshape(
        (nwalkers * (Nit - burn_in), ndim), order='c')
    lnpchain[:] = lnprobs[:, burn_in:].reshape((nwalkers * (Nit - burn_in)),
                                               order='c')

    ibestparams = np.argmax(lnpchain)
    bestparams = chains[ibestparams, :]

    iplot_burn_in = 0
    chains2 = np.zeros((Nit - iplot_burn_in, nwalkers, ndim))
    lnpchain2 = np.zeros(((Nit - iplot_burn_in), nwalkers))
    for j in list(range(nwalkers)):
        chains2[:, j, :] = samples[j, iplot_burn_in:, :].reshape(
            (Nit - iplot_burn_in, ndim), order='c')
        lnpchain2[:, j] = lnprobs[j, iplot_burn_in:].reshape(
            ((Nit - iplot_burn_in)), order='c')

    iplot_burn_in = burn_in
    chains3 = np.zeros((Nit - iplot_burn_in, nwalkers, ndim))
    lnpchain3 = np.zeros(((Nit - iplot_burn_in), nwalkers))
    for j in list(range(nwalkers)):
        chains3[:, j, :] = samples[j, iplot_burn_in:, :].reshape(
            (Nit - iplot_burn_in, ndim), order='c')
        lnpchain3[:, j] = lnprobs[j, iplot_burn_in:].reshape(
            ((Nit - iplot_burn_in)), order='c')

    ######### save bestparams
    if OptimM.Report:
        np.save(workdir + 'bestparams_mcmc.dat', bestparams)

    if OptimM.ChainPlots:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6.5, 15))
        par_labels = names
        for ip in range(nvar):
            ax = fig.add_subplot(nvar + 1, 1, ip + 1)
            ax.set_ylabel(par_labels[ip])
            ax.axes.xaxis.set_visible(False)
            for i in range(nwalkers):
                ax.plot(chains2[:, i, ip], alpha=0.1)

        ax2 = fig.add_subplot(nvar + 1, 1, nvar + 1)
        ax2.set_ylabel('ln(P)')
        ax2.set_xlabel('iter')
        for i in range(nwalkers):
            ax2.plot(lnpchain2[:, i], alpha=0.1)
        plt.savefig(workdir + 'chains.png', bbox_inches='tight')

        fig = plt.figure(figsize=(6.5, 15))
        par_labels = names
        for ip in range(nvar):
            ax = fig.add_subplot(nvar + 1, 1, ip + 1)
            ax.set_ylabel(par_labels[ip])
            ax.axes.xaxis.set_visible(False)
            for i in range(nwalkers):
                ax.plot(chains3[:, i, ip], alpha=0.1)

        ax2 = fig.add_subplot(nvar + 1, 1, nvar + 1)
        ax2.set_ylabel('ln(P)')
        ax2.set_xlabel('iter')
        for i in range(nwalkers):
            ax2.plot(lnpchain3[:, i], alpha=0.1)
        plt.savefig(workdir + 'chains_burn-in.png', bbox_inches='tight')

    #corner plots
    if OptimM.CornerPlots:
        cornerplotlabels = []
        for aname in names:
            if aname == 'log(Tdust)':
                cornerplotlabels.append(r'$\log(T_{\rm d})$')
            elif aname == 'log(amax)':
                cornerplotlabels.append(r'$\log(a_{\rm max})$')
            elif aname == 'log(Sigma_g)':
                cornerplotlabels.append(r'$\log(\Sigma_{\rm g})$')
            elif aname == 'q_dustexpo':
                cornerplotlabels.append(r'$q_{\rm dust}$')
            else:
                cornerplotlabels.append(aname)

        import corner
        print("calling corner")
        fig = corner.corner(
            chains,
            labels=cornerplotlabels,
            quantiles=[0.16, 0.5, 0.84],
            bins=40,  #truths=bestparams,
            levels=[0.68, 0.95, 0.997],
            #                  levels=3,
            show_titles=True,
            title_fmt=".2f",
            title_kwards={"fontsize": 6})  #, smooth=1.0

        fig.savefig(workdir + "triangle_all.png")

    mcmc_results = list(
        map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(chains, [16, 50, 84], axis=0))))

    # expectation values
    # ibestparams = np.argmax(lnpchain)
    # bestparams = chains[ibestparams, :]

    mcmc_results_0 = np.zeros(nvar)

    if OptimM.Report:
        print("param     distrib     max ")
    for iparam in list(range(nvar)):
        if OptimM.Report:
            print(names[iparam], mcmc_results[iparam], bestparams[iparam])
        mcmc_results_0[iparam] = mcmc_results[iparam][0]

    OptimM.mcmc_results = mcmc_results
    OptimM.mcmc_results_0 = mcmc_results_0
    OptimM.mcmc_bestparams = bestparams

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
        nchains_4plots = Nit - burn_in
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
                 chains=chains,
                 trueparams=OptimM.trueparams,
                 xscale=OptimM.summarySED_xaxisscale,
                 yscale=OptimM.summarySED_yaxisscale,
                 WithSEDchains=OptimM.summaryWithSEDchains,
                 nchains_4plots=nchains_4plots,
                 label4SED=OptimM.label4SED,
                 filename='fig_bestfit' + OptimM.filename_tag + '.png',
                 DoubleArrow=False)

    return [names, mcmc_results, bestparams, modelInus, modelalphas, achi2]
