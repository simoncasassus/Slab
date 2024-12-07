import sys
import os
import re
import numpy as np
import matplotlib
from copy import deepcopy
import matplotlib.pyplot as plt

from pprint import pprint

from astropy import constants as const
#from astropy.constants import astropyconst20 as const

from .Likelihood import assignfreeparams

from .AModelSED import MSED

c_MKS = const.c.value  # m/s
k_B = const.k_B.value

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/'
sys.path.append(include_path)


if not sys.warnoptions:
    import os, warnings
    #warnings.simplefilter("default") # Change the filter in this process
    warnings.simplefilter("ignore")  # Change the filter in this process
    #os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses



def plotSEDs(nvar,
                 names,
                 ASED,
                 ZData,
                 ZSetup,
                 CGbestparams=None,
                 mcmc_results=None,
                 mcmc_bestparams=None,
                 mcmc_results_0=None,
                 chains=None,
                 label4SED=None,
                 WithSEDchains=False,
                 nchains_4plots=1000,
                 PlotSpecIndexArrows=False,
                 scaleunits=1E6,
                 xscale='log',
                 yscale='log',
                 trueparams=False,
                 linearTd=True,
                 filename='fig_bestfit.png',
                 DoubleArrow=True):

    ZSetup4plots = deepcopy(ZSetup)
    ASED4plots = deepcopy(ASED)

    workdir = ZSetup4plots.outputdir
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

        UseCGBestFit = False
        if CGbestparams is not None:
            maxLvalueCG = CGbestparams[iparam]
            UseCGBestFit = True
            print("USING POWELL FOR MAX LIKELIHOOD VALUES")

            #if dolog:
            #    maxLvalueCG = 10**(maxLvalueCG)
            Ztitle_CG = Ztitle_CG + name + r'= %.2f ' % (maxLvalueCG)

        if mcmc_results is not None:

            parmcmc_results = mcmc_results[iparam]
            #parmcmc_results = mcmc_results_frombestparams[iparam]

            if UseCGBestFit:
                maxLvalue = maxLvalueCG
            else:
                maxLvalue = mcmc_bestparams[iparam]

            value = parmcmc_results[0]
            reportvalue = maxLvalue

            adjust_medianvalrange = False
            if adjust_medianvalrange:
                upperval = mcmc_results[iparam][0] + mcmc_results[iparam][1]
                uperror = upperval - maxLvalue
                lowerval = mcmc_results[iparam][0] - mcmc_results[iparam][2]
                downerror = maxLvalue - lowerval
            else:
                uperror = parmcmc_results[1]
                downerror = parmcmc_results[2]

            #if (uperror < 0.) | (downerror < 0.):
            #    reportvalue = parmcmc_results[0]
            #    uperror = parmcmc_results[1]
            #    downerror = parmcmc_results[2]

            #if dolog:
            #    uperror = 10**(value + uperror) - 10**(value)
            #    downerror = 10**(value) - 10**(value - downerror)
            #    value = 10**(value)
            #    maxLvalue = 10**(maxLvalue)
            Ztitle_maxL = Ztitle_maxL + name + r'= %.2f ' % (maxLvalue)
            #Ztitle_mcmc = Ztitle_mcmc + name + r'= %.2f$^{+%.2f}_{-%.2f}$ ' % (
            #    value, uperror, downerror)
            latexname = name
            if name == 'Tdust':
                latexname = r'$\log\left(\frac{T_{\rm d}}{{\rm K}}\right)$'
                latexname = r'$\log\left(T_{\rm d}/{\rm K}\right)$'
                if linearTd:
                    latexname = r'$T_{\rm d}$'
                    reportvalue = 10**reportvalue
                    uperror = np.log(10) * reportvalue * uperror
                    downerror = np.log(10) * reportvalue * downerror
                #latexname = ''
            if name == 'Sigma_g':
                latexname = r'$\log\left(\frac{\Sigma_{\rm g}}{{\rm g/cm}^2}\right)$'
                latexname = r'$\log\left(\Sigma_{\rm g}/{\rm g/cm}^2\right)$'
                #latexname = ''
            if name == 'amax':
                latexname = r'$\log\left(\frac{a_{\rm max}}{{\rm cm}}\right)$'
                latexname = r'$\log\left(a_{\rm max}/{\rm cm}\right)$'
                #latexname = r'$\log\left(a_{\rm max}/{\rm mm}\right)$'
                reportvalue = reportvalue

            if linearTd and (name == 'Tdust'):
                Ztitle_mcmc = Ztitle_mcmc + latexname + r'= %.1f$^{+%.1f}_{-%.1f}$  ' % (
                    reportvalue, uperror, downerror)
            else:
                Ztitle_mcmc = Ztitle_mcmc + latexname + r'= %.2f$^{+%.2f}_{-%.2f}$  ' % (
                    reportvalue, uperror, downerror)

            if linearTd and (name == 'Tdust'):
                Ztitle_mcmc += r'${\rm K}$'
        if name != 'Sigma_g':
            Ztitle_mcmc += '\n'

        if UseCGBestFit:
            Ztitle_CG = Ztitle_mcmc

    N_freqs = 100
    lognu1 = np.log10(90E9)
    lognu2 = np.log10(700E9)
    lognus = lognu1 + (np.arange(N_freqs) / N_freqs) * (lognu2 - lognu1)
    nus = 10**lognus

    # wavelengths must be in increasing order for dsharp_opac
    nus = np.flip(nus)

    ZSetup4plots.GoInterp = False
    ZSetup4plots.GoNearNeighbor1D = False
    ASED4plots.GoInterp = False
    ASED4plots.GoNearNeighbor1D = False
    ASED4plots.nus = nus
    ASED4plots.calcul(ForcePrep=True)

    Inorm = (ASED4plots.nus / 100E9)**2
    #plt.figure(figsize=(10, 4))
    #ax = plt.figure(figsize=(6, 3))
    ax = plt.figure(figsize=(4, 2))
    plt.ticklabel_format(axis='both', style='plain')

    if chains is not None:
        nchains, nvars = chains.shape
        print("nchains ", nchains, " nvars ", nvars)

    #ASED4plotsparams = MSED(ZSetup4plots)
    #ASED4plotsparams.copy(ASED4plots)  DEV
    ASED4plotsparams=deepcopy(ASED4plots)
    if WithSEDchains:
        print("plotting ", nchains_4plots, " sample SEDs")
        for ichain in range(nchains_4plots):
            #print("ichain", ichain)
            apars = chains[nchains - nchains_4plots + ichain, :]
            assignfreeparams(names, apars, ASED4plotsparams)
            ASED4plotsparams.calcul()
            plt.plot(ASED4plotsparams.nus / 1E9,
                     scaleunits * ZData.omega_beam * ASED4plotsparams.Inus /
                     Inorm,
                     alpha=0.002,
                     lw=0.5,
                     zorder=-32,
                     color='grey')

    if CGbestparams is not None:
        assignfreeparams(names, CGbestparams, ASED4plots)
        ASED4plots.calcul()
        plt.plot(
            ASED4plots.nus / 1E9,
            scaleunits * ZData.omega_beam * ASED4plots.Inus / Inorm,
            color='C3',
            zorder=2,
            #label=r'$I_\nu\, / \,(\nu/ \rm{100GHz})^2   $ maximum likelihood' +
            #'\n' + Ztitle_CG)
            #label=r'$I_\nu\, / \,(\nu/ \rm{100GHz})^2   $ maximum likelihood' +
            #'\n' + Ztitle_mcmc)
            #label=r'$I_\nu\, / \,(\nu/ \rm{100GHz})^2   $ ' +'\n' +  Ztitle_mcmc)
            label=Ztitle_CG)

        legendtext = Ztitle_CG

    elif mcmc_results:
        assignfreeparams(names, mcmc_bestparams, ASED4plots)
        ASED4plots.calcul()

        plt.plot(
            ASED4plots.nus / 1E9,
            scaleunits * ZData.omega_beam * ASED4plots.Inus / Inorm,
            color='C0',
            zorder=2,
            #label=r'$I_\nu\, / \,(\nu/ \rm{100GHz})^2   $ maximum likelihood' +
            #'\n' + Ztitle_mcmc)
            #label=r'$I_\nu\, / \,(\nu/ \rm{100GHz})^2   $ ' +'\n' +  Ztitle_mcmc)
            label=Ztitle_mcmc)

        legendtext = Ztitle_mcmc

        PlotMedianValues = False
        if PlotMedianValues:
            ASED4plotsmedian = ASED.MSED(ZSetup4plots)
            ASED4plotsmedian.copy(ASED4plots)
            assignfreeparams(names, mcmc_results_0, ASED4plotsmedian)
            ASED4plotsmedian.calcul()

            plt.plot(
                ASED4plotsmedian.nus / 1E9,
                scaleunits * ZData.omega_beam * ASED4plotsmedian.Inus / Inorm,
                color='C2',
                zorder=-10,
                label=r'$I_\nu\, / \,(\nu/ \rm{100GHz})^2   $ median' + '\n' +
                Ztitle_mcmc)

            # ensure numpy arrays are reset to mcmc_bestparams
            assignfreeparams(names, mcmc_bestparams, ASED4plots)
            ASED4plots.calcul()

    if trueparams:
        datalabel = ''  #'synthetic data for '+trueparams
    else:
        datalabel = ''

    Inorm_Data = (ZData.nus / 100E9)**2
    plt.errorbar(
        ZData.nus / 1E9,
        scaleunits * ZData.omega_beam * ZData.Inus / Inorm_Data,
        yerr=scaleunits * ZData.omega_beam * ZData.sInus / Inorm_Data,
        #yerr=scaleunits * ZData.omega_beam * 0.1 * ZData.Inus / Inorm_Data,
        color='C1',
        zorder=1,
        label=datalabel,
        elinewidth=3,
        linestyle='none',
        barsabove=True)

    plt.plot(ZData.nus / 1E9,
             scaleunits * ZData.omega_beam * ZData.Inus / Inorm_Data,
             color='C1',
             zorder=1,
             marker='x',
             linestyle='none')

    if trueparams:
        plt.title(trueparams, fontsize=8)

    if ZData.nus_alphas is not None:
        for ispecindex in range(len(ZData.alphas)):
            nu1 = ZData.nu1s_alphas[ispecindex]
            Inu1 = ZData.Inu1s[ispecindex]
            specindex = ZData.alphas[ispecindex]
            dnu = 0.1 * nu1
            nu2 = nu1 + dnu
            nu3 = nu1 - dnu
            #print("specindex", specindex)
            Inu2 = Inu1 * (nu2 / nu1)**specindex
            Inu3 = Inu1 * (nu3 / nu1)**specindex
            Inu1 /= (nu1 / 100E9)**2
            Inu2 /= (nu2 / 100E9)**2
            Inu3 /= (nu3 / 100E9)**2
            dInu = (Inu2 - Inu1)
            #print("Inu2", Inu2 * ZData.omega_beam, Inu1 * ZData.omega_beam)
            if PlotSpecIndexArrows:
                plt.arrow(nu1 / 1E9,
                          scaleunits * Inu1 * ZData.omega_beam,
                          dnu / 1E9,
                          scaleunits * dInu * ZData.omega_beam,
                          width=scaleunits * Inu1 * ZData.omega_beam * 0.01,
                          head_length=dnu / (2E9),
                          color='C1')
                if DoubleArrow:
                    dInu = (Inu3 - Inu1)
                    dnu = -0.1 * nu1
                    plt.arrow(nu1 / 1E9,
                              scaleunits * Inu1 * ZData.omega_beam,
                              dnu / 1E9,
                              scaleunits * dInu * ZData.omega_beam,
                              width=scaleunits * Inu1 * ZData.omega_beam *
                              0.01,
                              head_length=-dnu / (2E9),
                              color='C1')

    plt.ylabel(
        r'$I_\nu\, / \,\left(\frac{\nu}{\rm{100GHz}}\right)^2  ~~ / ~\mu$Jy beam$^{-1}$'
    )
    plt.xlabel(r'$\nu~~$ / GHz')
    print("xscale", xscale)
    print("yscale", yscale)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.legend(handlelength=0.)

    #plt.ticklabel_format(style='plain', axis='both')
    import matplotlib.ticker as mticker
    ax = plt.gca()
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.tick_params(which='both', axis='x', labelsize=8)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    #ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')
    plt.title(label4SED, fontsize=10)
    #xlabel=ax.get_xlim()[0]+0.005*(ax.get_xlim()[1]-ax.get_xlim()[0])
    #ylabel=ax.get_ylim()[1]-0.25*(ax.get_ylim()[1]-ax.get_ylim()[0])
    #if label4SED is None:
    #    label=''
    #else:
    #    label=label4SED
    #plt.text(xlabel, ylabel, label, ha='left', fontsize=10)

    #plt.ticklabel_format(axis='both',style='plain')
    #plt.grid()
    fileout = workdir + filename
    print(fileout)
    plt.savefig(fileout, bbox_inches='tight', dpi=500)
