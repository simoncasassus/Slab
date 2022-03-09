import sys
import os
import re
import numpy as np
import matplotlib
from time import time

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize as op
import cmath as cma

import emcee
import corner

from multiprocessing import Pool

from astropy import constants as const

c_MKS = const.c.value  # m/s

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/Slab/Continuum/'
sys.path.append(include_path)

import AModelSED

if not sys.warnoptions:
    import os, warnings
    #warnings.simplefilter("default") # Change the filter in this process
    warnings.simplefilter("ignore") # Change the filter in this process
    #os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

def summary_SEDs(nvar,
                 names,
                 ASED,
                 ZData,
                 ZSetup,
                 CGbestparams=None,
                 mcmc_results=None,
                 mcmc_bestparams=None,
                 mcmc_results_0=None,
                 chains=None,
                 nchains_4plots=1000,
                 filename='fig_bestfit.png',
                 DoubleArrow=True):
    workdir = ZSetup.outputdir
    Ztitle_maxL = ''
    Ztitle_CG = 'Powell '
    Ztitle_mcmc = ''
    for iparam in range(nvar):
        name = names[iparam]
        dolog = False
        m = re.search('log\((.*)\)', name)
        if m:
            dolog = True
            name = m.group(1)

        if CGbestparams is not None:
            maxLvalueCG = CGbestparams[iparam]
            if dolog:
                maxLvalueCG = 10**(maxLvalueCG)
            Ztitle_CG = Ztitle_CG + name + r'= %.1f ' % (maxLvalueCG)

        if mcmc_results is not None:
            parmcmc_results = mcmc_results[iparam]
            maxLvalue = mcmc_bestparams[iparam]
            value = parmcmc_results[0]
            uperror = parmcmc_results[1]
            downerror = parmcmc_results[2]
            if dolog:
                uperror = 10**(value + uperror) - 10**(value)
                downerror = 10**(value) - 10**(value - downerror)
                value = 10**(value)
                maxLvalue = 10**(maxLvalue)
            Ztitle_maxL = Ztitle_maxL + name + r'= %.1f ' % (maxLvalue)
            Ztitle_mcmc = Ztitle_mcmc + name + r'= %.1f+$^{+%.1f}_{-%.1f}$ ' % (
                value, uperror, downerror)

    N_freqs = 100
    lognu1 = np.log10(90E9)
    lognu2 = np.log10(700E9)
    lognus = lognu1 + (np.arange(N_freqs) / N_freqs) * (lognu2 - lognu1)
    nus = 10**lognus
    ASED.nus = nus
    ASED.calcul()

    Inorm = (ASED.nus / 100E9)**2
    plt.figure(figsize=(10, 4))
    WithSEDchains = False
    if chains is not None:
        WithSEDchains = True
        nchains, nvars = chains.shape
        print("nchains ", nchains, " nvars ", nvars)

    ASEDparams = AModelSED.MSED(ZSetup)
    ASEDparams.copy(ASED)
    if WithSEDchains:
        print("plotting ", nchains_4plots, " sample SEDs")
        for ichain in range(nchains_4plots):
            #print("ichain", ichain)
            apars = chains[nchains - nchains_4plots + ichain, :]
            assignfreeparams(names, apars, ASEDparams)
            ASEDparams.calcul()
            plt.plot(ASEDparams.nus / 1E9,
                     ZData.omega_beam * ASEDparams.Inus / Inorm,
                     alpha=0.01,
                     lw=0.5,
                     zorder=-32,
                     color='grey')

    if CGbestparams is not None:
        assignfreeparams(names, CGbestparams, ASED)
        ASED.calcul()
        plt.plot(
            ASED.nus / 1E9,
            ZData.omega_beam * ASED.Inus / Inorm,
            color='C3',
            zorder=2,
            label=r'$I_\nu\, / \,(\nu/ \rm{100GHz})^2   $ maximum likelihood' +
            '\n' + Ztitle_CG)

    if mcmc_results:
        assignfreeparams(names, mcmc_bestparams, ASED)
        ASED.calcul()

        plt.plot(
            ASED.nus / 1E9,
            ZData.omega_beam * ASED.Inus / Inorm,
            color='C0',
            zorder=2,
            label=r'$I_\nu\, / \,(\nu/ \rm{100GHz})^2   $ maximum likelihood' +
            '\n' + Ztitle_maxL)

        ASEDmedian = AModelSED.MSED(ZSetup)
        ASEDmedian.copy(ASED)
        assignfreeparams(names, mcmc_results_0, ASEDmedian)
        ASEDmedian.calcul()

        plt.plot(ASEDmedian.nus / 1E9,
                 ZData.omega_beam * ASEDmedian.Inus / Inorm,
                 color='C2',
                 zorder=2,
                 label=r'$I_\nu\, / \,(\nu/ \rm{100GHz})^2   $ median' + '\n' +
                 Ztitle_mcmc)

    Inorm_Data = (ZData.nus / 100E9)**2
    plt.errorbar(ZData.nus / 1E9,
                 ZData.omega_beam * ZData.Inus / Inorm_Data,
                 yerr=ZData.omega_beam * ZData.sInus / Inorm_Data,
                 color='C1',
                 zorder=1,
                 label='data',
                 elinewidth=3,
                 linestyle='none',
                 barsabove=True)

    plt.plot(ZData.nus / 1E9,
             ZData.omega_beam * ZData.Inus / Inorm_Data,
             color='C1',
             zorder=1,
             marker='x',
             linestyle='none')

    if ZData.nus_alphas is not None:
        for ispecindex in range(len(ZData.alphas)):
            nu1 = ZData.nu1s_alphas[ispecindex]
            Inu1 = ZData.Inu1s[ispecindex]
            specindex = ZData.alphas[ispecindex]
            dnu = 0.1 * nu1
            nu2 = nu1 + dnu
            nu3 = nu1 - dnu
            print("specindex", specindex)
            Inu2 = Inu1 * (nu2 / nu1)**specindex
            Inu3 = Inu1 * (nu3 / nu1)**specindex
            Inu1 /= (nu1 / 100E9)**2
            Inu2 /= (nu2 / 100E9)**2
            Inu3 /= (nu3 / 100E9)**2
            dInu = (Inu2 - Inu1)
            print("Inu2", Inu2 * ZData.omega_beam, Inu1 * ZData.omega_beam)
            plt.arrow(nu1 / 1E9,
                      Inu1 * ZData.omega_beam,
                      dnu / 1E9,
                      dInu * ZData.omega_beam,
                      width=Inu1 * ZData.omega_beam * 0.01,
                      head_length=dnu / (2E9),
                      color='C1')
            if DoubleArrow:
                dInu = (Inu3 - Inu1)
                dnu = -0.1 * nu1
                plt.arrow(nu1 / 1E9,
                          Inu1 * ZData.omega_beam,
                          dnu / 1E9,
                          dInu * ZData.omega_beam,
                          width=Inu1 * ZData.omega_beam * 0.01,
                          head_length=-dnu / (2E9),
                          color='C1')

    plt.ylabel(r'Jy beam$^{-1}$')
    plt.xlabel(r'$\nu$ / GHz')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    fileout = workdir + filename
    print(fileout)
    plt.savefig(fileout, bbox_inches='tight')


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


def lnlike(x_free, parnames, ZSetup, ZData, ASED, ZMerit):

    assignfreeparams(parnames, x_free, ASED)

    
    chi2 = ZMerit.calcul(ZSetup, ZData, ASED)
    chi2 = chi2.item()
    # if OptimM.PrintChi2s:
    # print("lnlike chi2 %e " % (chi2))

    return -0.5 * chi2


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


def lnprob(x_free, parnames, bnds, ZSetup, ZData, ASED, ZMerit):
    lp = lnprior(x_free, bnds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(x_free, parnames, ZSetup, ZData, ASED, ZMerit)


def run_scipy_optimize_minimize(x_free, names, bnds, ZSetup, ZData, ASED,
                                ZMerit, OptimM):
    #print("starting op.minimize")
    start_time = time()
    ftol = 1E-10  # 1e-10 too small leads to abnormal termination
    eps = []
    for ibnd, abnd in enumerate(bnds):
        if ((x_free[ibnd] < abnd[0]) or (x_free[ibnd] > abnd[1])):
            print("param ", names[ibnd], " is out of range")
            sys.exit()
        eps.append((abnd[1] - abnd[0]) / 100.)
    eps = np.array(eps)
    nll = lambda *args: -lnlike(*args)
    x_free_init = x_free.copy()

    init_lnlike = lnlike(x_free_init, names, ZSetup, ZData, ASED, ZMerit)
    if OptimM.Report:
        print("init_chi2 %e" % (-2. * init_lnlike))
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
        args=(names, ZSetup, ZData, ASED, ZMerit))
    result_ml = result["x"]
    if OptimM.Report:
        print("result", result)
        print("Optim done in (elapsed time):", time() - start_time)
        print("Delta params", result_ml - x_free_init)

    init_lnlike2 = lnlike(x_free_init, names, ZSetup, ZData, ASED, ZMerit)
    if OptimM.Report:
        print("init chi2 xcheck   %e" % (-2. * init_lnlike2))
    best_lnlike = lnlike(result_ml, names, ZSetup, ZData, ASED, ZMerit)
    if OptimM.Report:
        print("best_chi2 %e" % (-2. * best_lnlike))

    if OptimM.Report:
        print("computing errors with Hessian")

    tmp_i = np.zeros(len(result_ml))
    errors_ml = np.zeros(len(result_ml))
    #for i in list(range(len(result_ml))):
    #    tmp_i[i] = 1.0
    #    uncertainty_i = np.sqrt(result.hess_inv(tmp_i)[i])
    #    errors_ml[i] = uncertainty_i
    #    # print(('{0:12.4e} +- {1:.1e}'.format(result.x[i], uncertainty_i)))
    return (result_ml, errors_ml)


def exec_ConjGrad(OptimM, ZSetup, ZData, ASED, ZMerit):

    maxiter = OptimM.CGmaxiter
    ASED.nus = ZData.nus

    names = list(map((lambda x: x[0]), OptimM.domain))
    sample_params = list(map((lambda x: x[1]), OptimM.domain))
    bnds = list(map((lambda x: x[2]), OptimM.domain))
    nvar = len(list(names))

    if OptimM.Inherit_Init:
        for iname, aname in enumerate(names):
            aname_4value=aname
            m = re.search('log\((.*)\)', aname)
            dolog=False
            if m:
                dolog = True
                aname_4value = m.group(1)
            value=getattr(ASED, aname_4value)
            if dolog:
                value=np.log10(value)
            sample_params[iname] = value
            if OptimM.Report:
                print("Inherit_Init: ", aname, sample_params[iname])


    if OptimM.Report:
        print("nvar = ", nvar)

    x_free = np.array(sample_params)

    (result_ml,
     errors_ml) = run_scipy_optimize_minimize(x_free, names, bnds, ZSetup,
                                              ZData, ASED, ZMerit, OptimM)

    if OptimM.Report:
        np.save(ZSetup.outputdir + 'result_ml.dat', result_ml)
        np.save(ZSetup.outputdir + 'result_ml_errors.dat', errors_ml)
    if OptimM.SummaryPlots:

        summary_SEDs(nvar,
                     names,
                     ASED,
                     ZData,
                     ZSetup,
                     CGbestparams=result_ml,
                     mcmc_results=None,
                     mcmc_bestparams=None,
                     mcmc_results_0=None,
                     chains=None,
                     nchains_4plots=False,
                     filename='fig_bestfit_Powell.png',
                     DoubleArrow=False)

    return names, result_ml




def exec_emcee(OptimM, ZSetup, ZData, ASED, ZMerit):
    # Inherit_Init: inherit initial conditions from ASED rather than from OptimM.domain

    ASED.nus = ZData.nus

    workdir = ZSetup.outputdir

    names = list(map((lambda x: x[0]), OptimM.domain))
    sample_params = list(map((lambda x: x[1]), OptimM.domain))
    bnds = list(map((lambda x: x[2]), OptimM.domain))
    nvar = len(list(names))

    if OptimM.Inherit_Init:
        for iname, aname in enumerate(names):
            aname_4value=aname
            m = re.search('log\((.*)\)', aname)
            dolog=False
            if m:
                dolog = True
                aname_4value = m.group(1)
            value=getattr(ASED, aname_4value)
            if dolog:
                value=np.log10(value)
            sample_params[iname] = value
            if OptimM.Report:
                print("Inherit_Init: ", aname, sample_params[iname])

    if OptimM.Report:
        print("nvar = ", nvar)

    x_free = np.array(sample_params)
    x_free_init = x_free.copy()

    if OptimM.Report:
        print("check initial chi2 value")
    init_lnlike = lnlike(x_free, names, ZSetup, ZData, ASED, ZMerit)
    if OptimM.Report:
        print("init lnlike  = %e " % (init_lnlike))
        print("chi2 = %e " % (-2 * init_lnlike))

    #sys.exit()

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
        awalkerinit = sample_params + (1e-2 * np.random.randn(ndim) *
                                       allowed_ranges)

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
        lnprobargs = [names, bnds, ZSetup, ZData, ASED, ZMerit]
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
                ax.plot(chains2[:, i, ip], alpha=0.01)

        ax2 = fig.add_subplot(nvar + 1, 1, nvar + 1)
        ax2.set_ylabel('ln(P)')
        ax2.set_xlabel('iter')
        for i in range(nwalkers):
            ax2.plot(lnpchain2[:, i], alpha=0.01)
        plt.savefig(workdir + 'chains.png', bbox_inches='tight')

    #corner plots
    if OptimM.CornerPlots:
        import corner
        print("calling corner")
        fig = corner.corner(
            chains,
            labels=names,
            quantiles=[0.16, 0.5, 0.84],
            bins=40,  #truths=bestparams,
            levels=[0.68, 0.95, 0.997],
            #                  levels=3,
            show_titles=True,
            title_fmt=".1f",
            title_kwards={"fontsize": 6})  #, smooth=1.0

        fig.savefig(workdir + "triangle_all.png")

    mcmc_results = list(
        map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(chains, [16, 50, 84], axis=0))))

    mcmc_results_0 = np.zeros(nvar)

    if OptimM.Report:
        print("param     distrib     max ")
    for iparam in list(range(nvar)):
        if OptimM.Report:
            print(names[iparam], mcmc_results[iparam], bestparams[iparam])
        mcmc_results_0[iparam] = mcmc_results[iparam][0]

    OptimM.mcmc_results = mcmc_results
    OptimM.mcmc_bestparams = bestparams

    if OptimM.Report:
        print("cross check initial chi2 value")
        init_lnlike = lnlike(x_free_init, names, ZSetup, ZData, ASED, ZMerit)
        print("chi2 = %e " % (-2 * init_lnlike))

    if OptimM.Report:
        print("running final lnlike to set model to median values")
        final_lnlike = lnlike(mcmc_results_0, names, ZSetup, ZData, ASED,
                              ZMerit)
        print("chi2 = %e " % (-2 * final_lnlike))

    if OptimM.Report:
        print("running final lnlike to set model to best L  values")
        ZSetup.filetag = 'mcmc_bestL_'
    final_lnlike = lnlike(bestparams, names, ZSetup, ZData, ASED, ZMerit)
    modelInus=ASED.Inus
    modelalphas=ASED.alphas
    achi2=-2 * final_lnlike
    if OptimM.Report:
        print("chi2 = %e" % (achi2x))

    ######################################################################
    # plot results

    if OptimM.SummaryPlots:
        nchains_4plots = Nit - burn_in
        #if (nchains_4plots > 1000):
        #    nchains_4plots=1000

        summary_SEDs(nvar,
                     names,
                     ASED,
                     ZData,
                     ZSetup,
                     CGbestparams=None,
                     mcmc_results=mcmc_results,
                     mcmc_bestparams=bestparams,
                     mcmc_results_0=mcmc_results_0,
                     chains=chains,
                     nchains_4plots=nchains_4plots,
                     DoubleArrow=False)

    return [names, mcmc_results, bestparams, modelInus, modelalphas, chi2]


def logL(ZData, ASED, ASED4alphas=False):

    if (np.sum(ZData.sInus) == 0):
        chi2 = np.array([0.])
    else:
        chi2 = np.sum((ZData.Inus - ASED.Inus)**2 / ZData.sInus**2)

    # print("chi2  = ", chi2, " dofs ", len(ZData.Inus))
    if ASED4alphas:
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
    # print("chi2  = ", chi2)
    return chi2


class Data():

    def __init__(
            self,
            file_obsInus='',
            file_obsalphas=False,  # ''
            VerboseInit=False,
            ######################################################################
            nus=None,
            Inus=None,
            sInus=None,  # standard deviation error 
            nu1s_alphas=None,
            Inu1s=None,
            omega_beam=(np.pi / (4. * np.log(2))) *
            (0.040 * np.pi / (180. * 3600.))**2,  # C10 B3 beam
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

    def copy(self, AnotherData):
        self.__dict__.update(AnotherData.__dict__)


class Merit():

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

    # self.__dict__.update(ASED.__dict__)

    def calcul(self, ZSetup, ZData, ASED):
        if self.ExecTimeReport:
            time_start = time()

        ASED.calcul()
        if self.ExecTimeReport:
            time_end_1 = time()
            print("time for ASED calcul: ", time_end_1 - time_start, " s")
        ASED4alphas = False
        if ZData.nus_alphas is not None:
            ASED4alphas = AModelSED.MSED(ZSetup)
            ASED4alphas.copy(ASED)
            ASED4alphas.nus = ZData.nus_alphas
            ASED4alphas.calcul()
            if self.ExecTimeReport:
                time_end_2 = time()
                print("time for ASED4alphas calcul: ", time_end_2 - time_end_1,
                      " s")

        chi2 = logL(ZData, ASED, ASED4alphas=ASED4alphas)

        if self.ExecTimeReport:
            time_end = time()
            print("total time for Likelihood evaluation: ",
                  time_end - time_start, " s")

        return chi2


class OptimM():

    def __init__(
            self,
            RunMCMC=False,
            RunConjGrad=True,
            CGmaxiter=100,
            MCMC_Nit=100,  #MCMC iterations
            nwalkers_pervar=10,
            burn_in=50,
            n_cores_MCMC=4,
            CornerPlots=True,  # TriangleFile='cornerplot.png',
            ChainPlots=True,
            PrintChi2s=False,
            Report=True,
            SummaryPlots=True,
            domain=[],
            domain_CG=[],
            domain_MCMC=[],
            Inherit_Init=False, # loads init conditions from ASED 
            MCMCProgress=True,
            ######################################################################
            mcmc_results=[],
            mcmc_bestparams=[]):

        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            print("setting ", a_attribute, " to ", initlocals[a_attribute])
            setattr(self, a_attribute, initlocals[a_attribute])

    def MCMC(self, ZSetup, ZData, ASED, ZMerit):
        [names, mcmc_results, bestparams,  modelInus, modelalphas, achi2] = exec_emcee(self, ZSetup, ZData, ASED, ZMerit)
        return [names, mcmc_results, bestparams,  modelInus, modelalphas]

    def ConjGrad(self, ZSetup, ZData, ASED, ZMerit):
        return exec_ConjGrad(self, ZSetup, ZData, ASED, ZMerit)
