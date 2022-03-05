import sys
import os
import re
import numpy as np
import matplotlib
from time import time

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma

import emcee
import corner


from astropy import constants as const

c_MKS = const.c.value  # m/s

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/Slab/Continuum/'
sys.path.append(include_path)

import AModelSED


def assignfreeparams(parnames, values, ASED):
    
    for iparam,aparam in enumerate(parnames):
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


def exec_emcee(OptimM, ZSetup, ZData, ASED, ZMerit):

    workdir = ZSetup.outputdir

    names = list(map((lambda x: x[0]), OptimM.domain))
    sample_params = list(map((lambda x: x[1]), OptimM.domain))
    bnds = list(map((lambda x: x[2]), OptimM.domain))
    nvar = len(list(names))

    print("nvar = ", nvar)

    x_free = np.array(sample_params)
    x_free_init = x_free.copy()

    print("check initial chi2 value")
    init_lnlike = lnlike(x_free, names, ZSetup, ZData, ASED, ZMerit)
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
        awalkerinit = sample_params + (1e-3 * np.random.randn(ndim) *
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

    print("in exec_emcee with RunMCMC=", OptimM.RunMCMC)
    if OptimM.RunMCMC:
        print("now about to call run_mcmc with Nit", Nit, "and nmwalkers",
              nwalkers, " and ncores", n_cores)

        from multiprocessing import Pool
        with Pool(n_cores) as pool:
            lnprobargs = [names, bnds,  ZSetup, ZData, ASED, ZMerit]

            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            lnprob,
                                            args=lnprobargs,
                                            pool=pool)
            start = time()
            sampler.run_mcmc(pos, Nit, progress=True)
            end = time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        
        samples = sampler.chain  # chain= array(nwalkers,nit,ndim)
        lnprobs = sampler.lnprobability

        ######### save samples
        np.save(workdir + 'samples.dat', samples)
        np.save(workdir + 'lnprobs.dat', lnprobs)
        Delta_t = multi_time / 60.
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
        plt.savefig(workdir + 'chains.png',
                    bbox_inches='tight')

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

    print("param     distrib     max ")
    for iparam in list(range(nvar)):
        print(names[iparam], mcmc_results[iparam], bestparams[iparam])
        mcmc_results_0[iparam] = mcmc_results[iparam][0]

    OptimM.mcmc_results = mcmc_results
    OptimM.mcmc_bestparams = bestparams

    print("cross check initial chi2 value")
    init_lnlike = lnlike(x_free_init, names, ZSetup, ZData, ASED, ZMerit)
    print("chi2 = %e " % (-2 * init_lnlike))
    
    print("running final lnlike to set model to mean values")
    final_lnlike = lnlike(mcmc_results_0, names, ZSetup, ZData, ASED, ZMerit)
    print("chi2 = %e " % (-2 * final_lnlike))
    
    print("running final lnlike to set model to best L  values")
    ZSetup.filetag = 'mcmc_bestL_'
    final_lnlike = lnlike(bestparams, names, ZSetup, ZData, ASED, ZMerit)
    print("chi2 = %e" % (-2 * final_lnlike))

    N_freqs = 100
    lognu1 = np.log10(30E9)
    lognu2 = np.log10(700E9)
    lognus = lognu1 + (np.arange(N_freqs) / N_freqs) * (lognu2 - lognu1)
    nus = 10**lognus
    ASED.nus=nus
    ASED.calcul()

    Inorm=(ASED.nus/ZData.nus[0])**2
    plt.figure(figsize=(10, 4))
    plt.plot(ASED.nus / 1E9, ASED.Inus/Inorm, label=r'$I_\nu / (\nu/\nu_\circ)^2 $')
    Inorm_Data=(ZData.nus/ZData.nus[0])**2
    plt.errorbar(ZData.nus/1E9,ZData.Inus/Inorm_Data,yerr=ZData.sInus,label='data',linestyle='none')
    plt.ylabel('Jy/sr')
    plt.xlabel(r'$\nu$ / GHz')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    fileout = workdir + 'fig_bestfit.png'
    plt.savefig(fileout, bbox_inches='tight')



    
    return [names, mcmc_results]


def logL(ZData, ASED, ASED4alphas=False):

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
        self.nus, self.Inus, self.sInus = np.loadtxt(self.file_obsInus,
                                                     unpack=True)
        if self.file_obsalphas:
            self.nu1s_alphas, self.nu2s_alphas, self.alphas, self.salphas = np.loadtxt(
                self.file_obsalphas, unpack=True)
            npairs = len(self.nu1s_alphas)
            allnus = np.zeros(int(2 * npairs))
            for ipair in range(npairs):
                inu1 = int(2 * ipair)
                inu2 = int(2 * ipair + 1)
                allnus[inu1] = self.nu1s_alphas[ipair]
                allnus[inu2] = self.nu2s_alphas[ipair]
            self.nus_alphas = allnus


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
            print("time for ASED calcul: ", time_end_1 - time_start," s")
        ASED4alphas = False
        if ZData.file_obsalphas:
            ASED4alphas = AModelSED.MSED(ZSetup)
            ASED4alphas.copy(ASED)
            ASED4alphas.nus = ZData.nus_alphas
            ASED4alphas.calcul()
            if self.ExecTimeReport:
                time_end_2 = time() 
                print("time for ASED4alphas calcul: ", time_end_2 - time_end_1," s")

        chi2 = logL(ZData, ASED, ASED4alphas=ASED4alphas)

        if self.ExecTimeReport:
            time_end = time()
            print("total time for Likelihood evaluation: ", time_end - time_start,
                  " s")

        return chi2


class OptimM():

    def __init__(
            self,
            RunMCMC=False,
            MCMC_Nit=100,  #MCMC iterations
            nwalkers_pervar=10,
            burn_in=50,
            n_cores_MCMC=4,
            CornerPlots=True,  # TriangleFile='cornerplot.png',
            ChainPlots=True,
            PrintChi2s=False,
            domain=[],
            ######################################################################
            mcmc_results=[],
            mcmc_bestparams=[]):

        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            print("setting ", a_attribute, " to ", initlocals[a_attribute])
            setattr(self, a_attribute, initlocals[a_attribute])

    def MCMC(self, ZSetup, ZData, ASED, ZMerit):
        [names, mcmc_results] = exec_emcee(self, ZSetup, ZData, ASED, ZMerit)
        return [names, mcmc_results]
