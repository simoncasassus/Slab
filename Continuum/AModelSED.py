import sys
import os
import re
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
from time import time
from astropy import constants as const

# matplotlib.use('Agg')

from numba import jit

c_MKS = const.c.value  # m/s


def Plot_Opct(Opct, outputdir):
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(Opct[:, 0], Opct[:, 1])
    plt.ylabel('n')
    plt.xlabel(r'$\lambda$ [$\mu$m]')

    plt.xscale('log')
    plt.subplot(122)
    plt.plot(Opct[:, 0], Opct[:, 2])
    plt.ylabel('k')
    plt.xlabel(r'$\lambda$ [$\mu$m]')

    plt.xscale('log')
    plt.yscale('log')

    fileout = 'fig_optconstants.pdf'
    plt.savefig(outputdir + fileout, bbox_inches='tight')


def Plot_Inu(nus, Inus, overplots=[], outputdir='', fileout='fig_Inu.pdf'):
    plt.figure(figsize=(10, 4))
    for iover, aover in enumerate(overplots):
        anus = aover[0]
        aInus = aover[1]
        label = aover[2]
        plt.plot(anus / 1E9, aInus, label=label)
    plt.plot(nus / 1E9, Inus, label=r'$I_\nu$')
    #plt.plot(nus / 1E9, Inus[-1] * (nus / nus[-1])**2, label=r'$\nu^2$')
    #plt.ylabel('Jy/sr')
    plt.ylabel('Jy/beam')
    #plt.ylabel('Jy/brick')
    plt.xlabel(r'$\nu$ / GHz')
    #plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    fileout = outputdir + fileout
    plt.savefig(fileout, bbox_inches='tight')


def Bnu_Jy(freq, T):

    #     inputs:
    #      frequency  / Hz
    #      T  / K
    #     output:
    #      planck radiation in Jy/sr

    nu = freq
    bnu = 1.4745E-50 * (nu**3) / (np.exp(4.79922E-11 * nu / T) - 1.0)
    return (bnu * 1E26)


# Opacity function
def f_kappa_abs(nf, kf, a, lam, rho, f):
    # Kataoka+2014
    lamc = 2. * np.pi * a

    m = nf(lam * 1.0e4) + kf(lam * 1.0e4) * 1.j
    eps0 = m**2.0
    F = (eps0 - 1.) / (eps0 + 2.)
    eps_ef = (1. + 2. * f * F) / (1. - f * F)
    mef = cma.sqrt(eps_ef)
    nef = mef.real
    kef = mef.imag

    x = lamc / lam
    if x < 1.0:
        Qabs = 24. * nef * kef * x / ((nef**2. - kef**2. + 2.)**2. +
                                      (2. * nef * kef)**2.)
    else:  #kef*x<3./8.:
        Qabs2 = 8. * kef * x * (nef**3. - (nef**2 - 1.)**1.5) / (3. * nef)
        Qabs3 = 1. - 0.1 * f
        Qabs = min(Qabs2, Qabs3)
    #print x, lam
    kappa0 = 3. / (4. * a * rho)
    return kappa0 * Qabs


# Opacity function


@jit(nopython=True)
def f_kappa_abs_numba(nf, kf, a, lam, rho, f):
    # Kataoka+2014
    lamc = 2. * np.pi * a

    m = nf + kf * 1.j
    eps0 = m**2.0
    F = (eps0 - 1.) / (eps0 + 2.)
    eps_ef = (1. + 2. * f * F) / (1. - f * F)
    mef = cma.sqrt(eps_ef)
    nef = mef.real
    kef = mef.imag

    x = lamc / lam
    if x < 1.0:
        Qabs = 24. * nef * kef * x / ((nef**2. - kef**2. + 2.)**2. +
                                      (2. * nef * kef)**2.)
    else:
        Qabs2 = 8. * kef * x * (nef**3. - (nef**2 - 1.)**1.5) / (3. * nef)
        Qabs3 = 1. - 0.1 * f
        Qabs = min(Qabs2, Qabs3)

    kappa0 = 3. / (4. * a * rho)
    return kappa0 * Qabs


@jit(nopython=True)
def f_kappa_scat_numba(nf, kf, a, lam, rho, f):
    # Kataoka+2014
    lamc = 2. * np.pi * a

    m = nf + kf * 1.j
    eps0 = m**2.0
    F = (eps0 - 1.) / (eps0 + 2.)
    eps_ef = (1. + 2. * f * F) / (1. - f * F)
    mef = cma.sqrt(eps_ef)
    nef = mef.real
    kef = mef.imag

    x = lamc / lam
    Qscat1 = (32. / 27.) * (x**4.) * ((nef - 1.)**2. + kef**2.)
    if x < 1.0:
        Qscat = Qscat1
    else:  #kef*x<3./8.:
        Qscat2 = Qscat1 / (x**2.)
        Qscat3 = 1. + 0.1 * f
        Qscat = min(Qscat2, Qscat3)
    #print x, lam
    kappa0 = 3. / (4. * a * rho)
    return kappa0 * Qscat


@jit(nopython=True)
def get_kappa_as_numba_kernel(nlambdas, lambdas, nfs, kfs, rhoi, N_asizes,
                              a_sizes, kappa_as_abs, kappa_as_scat, f_grain):

    for j in range(nlambdas):
        for i_asize in range(N_asizes):
            a_asize = a_sizes[i_asize]
            kappa_as_abs[i_asize,
                         j] = f_kappa_abs_numba(nfs[j], kfs[j], a_asize,
                                                lambdas[j], rhoi, f_grain)
            kappa_as_scat[i_asize,
                          j] = f_kappa_scat_numba(nfs[j], kfs[j], a_asize,
                                                  lambdas[j], rhoi, f_grain)


def f_kappa_scat(nf, kf, a, lam, rho, f):
    # Kataoka+2014
    lamc = 2. * np.pi * a

    m = nf(lam * 1.0e4) + kf(lam * 1.0e4) * 1.j
    eps0 = m**2.0
    F = (eps0 - 1.) / (eps0 + 2.)
    eps_ef = (1. + 2. * f * F) / (1. - f * F)
    mef = cma.sqrt(eps_ef)
    nef = mef.real
    kef = mef.imag

    x = lamc / lam
    Qscat1 = (32. / 27.) * (x**4.) * ((nef - 1.)**2. + kef**2.)
    if x < 1.0:
        Qscat = Qscat1
    else:  #kef*x<3./8.:
        Qscat2 = Qscat1 / (x**2.)
        Qscat3 = 1. + 0.1 * f
        Qscat = min(Qscat2, Qscat3)
    #print x, lam
    kappa0 = 3. / (4. * a * rho)
    return kappa0 * Qscat


def Inu_Bnu_unifslab_direct(tau_nu_total, epsilon_nu):

    tau = tau_nu_total

    tmpA = (1. / ((np.exp(-np.sqrt(3. * epsilon_nu) * tau) *
                   (np.sqrt(epsilon_nu) - 1.)) - (np.sqrt(epsilon_nu) + 1.)))

    F1 = ((1. - np.exp(-(np.sqrt(3. * epsilon_nu) + 1) * tau)) /
          (1. + np.sqrt(3. * epsilon_nu)))
    F2 = np.exp(-np.sqrt(3. * epsilon_nu) * tau) * (np.exp(
        (np.sqrt(3. * epsilon_nu) - 1.) * tau) - 1.) / (
            np.sqrt(3. * epsilon_nu) - 1.)

    F = (F1 + F2) * tmpA

    Inu_Bnu = (1. - np.exp(-tau)) + (1. - epsilon_nu) * F

    return Inu_Bnu


class Setup():

    def prep(self):
        if (not re.search(r"\/$", self.outputdir)):
            self.outputdir += '/'
            print("added trailing back slash to outputdir")
        if ((os.path.exists(self.outputdir) & (self.ClearOutputDir))):
            os.system("rm -rf " + self.outputdir)
        os.system("mkdir " + self.outputdir)

    def load_Opct(self):
        Opct = np.loadtxt(
            self.opct_file)  # mix of silicates, amorphous carbon and water ice
        if self.GenFigs:
            Plot_Opct(Opct, self.outputdir)
        self.nf = interp1d(Opct[:, 0], Opct[:, 1], kind='cubic')
        self.kf = interp1d(Opct[:, 0], Opct[:, 2], kind='cubic')

    def __init__(
            self,
            filetag='',  # False
            Verbose=True,
            PrintChi2s=True,
            ClearOutputDir=False,
            nf=None,
            kf=None,
            #log_a_sizes_4interp=None,
            GoInterp=False,
            amax_4grid=10,  #cm
            GenFigs=True,
            opct_file='opct_mix.txt',
            VerboseInit=False,
            use_Kataoka=True,  # 4 grids
            outputdir='./output_dev/',
            ######################################################################
            kappa_as_abs_4interp=None,
            kappa_as_scat_4interp=None):

        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            if VerboseInit:
                print("Setup> setting ", a_attribute, " to ",
                      initlocals[a_attribute])
            setattr(self, a_attribute, initlocals[a_attribute])

        self.prep()
        self.load_Opct()


class MSED(Setup):

    def __init__(
            self,
            ASetup,
            Tdust=30.,
            q_dustexpo=-3.5,
            f_grain=1.,  # grain filling factor
            amin=1E-4,  # cm
            amax=1.,  # cm, maximum grain size
            Sigma_g=0.5,  # g/cm2
            gtod_ratio=100.,
            rho0=2.77,  # g/cm3
            N_asizes=40,  #if ZSetup.GoInterp  then N_asizes should be set to the grid
            nus=[],
            ExecTimeReport=False,
            GoNumba=False,
            ######################################################################
            Inus=[],
            Sigma_d=0,
            a_sizes=[],
            Sigma_as=[],
            lambdas=[],
            N_freqs=0,
            kappa_as_abs=None,
            kappa_as_scat=None,
            ilogamax=None,
            tau_abs=None,
            tau_scat=None,
            tau=None,
            kappa_abs=None,
            kappa_scat=None):

        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            if ASetup.VerboseInit:
                print("MSED> setting ", a_attribute, " to ",
                      initlocals[a_attribute])
            setattr(self, a_attribute, initlocals[a_attribute])

        self.__dict__.update(ASetup.__dict__)

        if self.GoInterp:
            if self.kappa_as_abs_4interp is None:
                GoNumba = self.GoNumba
                if ASetup.use_Kataoka:
                    print("computing opacity grid using Kataoka formulae")
                    self.GoNumba = True
                    self.GoInterp = False
                    amax=self.amax
                    self.amax = self.amax_4grid
                    self.calcul()
                    ASetup.kappa_as_abs_4interp = self.kappa_as_abs
                    ASetup.kappa_as_scat_4interp = self.kappa_as_scat
                    self.kappa_as_abs_4interp = self.kappa_as_abs
                    self.kappa_as_scat_4interp = self.kappa_as_scat
                    self.GoNumba = GoNumba
                    self.GoInterp = True
                    self.amax = amax
                else:
                    if os.path.exists(outputdir + 'allkappa_as_scat.npy'):
                        self.kappa_as_abs_4interp = np.load(
                            self.outputdir + 'allkappa_as_abs.npy')
                        self.kappa_as_scat_4interp = np.load(
                            self.outputdir + 'allkappa_as_scat.npy')
                        self.log_a_sizes_4interp = np.load(outputdir +
                                                           'logasizes.npy')
                    else:
                        sys.exit("compute grids first")

    def prep(self):
        self.N_freqs = len(self.nus)
        self.Sigma_d = self.Sigma_g / self.gtod_ratio
        #self.a_sizes = np.logspace(np.log10(self.amin), np.log10(self.amax),
        #                           self.N_asizes)

        if not self.GoInterp:
            loga_sizes = (np.log10(self.amax) - np.log10(self.amin)) * (
                np.arange(self.N_asizes) / self.N_asizes) + np.log10(self.amin)
            self.a_sizes = 10**(loga_sizes)
            

    def copy(self, AnotherSED):
        self.__dict__.update(AnotherSED.__dict__)

    def get_Sigma_as(self): DEV DEV DEV 
        fas = np.zeros(
            self.N_asizes)  # Surface density for different dust sizes
        for i in range(self.N_asizes):
            if i == 0:
                fas[i] = self.a_sizes[i]**(self.q_dustexpo + 4.)
            else:
                fas[i] = self.a_sizes[i]**(self.q_dustexpo +
                                           4.) - self.a_sizes[i - 1]**(
                                               self.q_dustexpo + 4.)

        self.Sigma_as = fas * self.Sigma_d / np.sum(fas)

        self.lambdas = 1E2 * c_MKS / self.nus  # wavelengths in cm

    def get_kappa_as(self):
        kappa_as_abs = np.zeros(
            (self.N_asizes, self.N_freqs
             ))  # opacities for different grain sizes and wavelengths
        kappa_as_scat = np.zeros(
            (self.N_asizes, self.N_freqs
             ))  # opacities for different grain sizes and wavelengths

        rhoi = self.rho0 * self.f_grain
        for i_asize, a_asize in enumerate(self.a_sizes):
            for j in range(len(self.lambdas)):
                kappa_as_abs[i_asize,
                             j] = f_kappa_abs(self.nf, self.kf, a_asize,
                                              self.lambdas[j], rhoi,
                                              self.f_grain)
                kappa_as_scat[i_asize,
                              j] = f_kappa_scat(self.nf, self.kf, a_asize,
                                                self.lambdas[j], rhoi,
                                                self.f_grain)
        self.kappa_as_abs = kappa_as_abs
        self.kappa_as_scat = kappa_as_scat

    def get_kappa_as_interp(self):

        logamax = np.log10(self.amax)
        dlogas = (np.log10(self.amax_4grid) -
                  np.log10(self.amin)) / self.N_asizes
        ilogamax = int((logamax - np.log10(self.amin)) / dlogas)

        self.kappa_as_abs = self.kappa_as_abs_4interp
        self.kappa_as_scat = self.kappa_as_scat_4interp
        self.ilogamax = ilogamax + 1

    def get_kappa_as_numba(self):

        kappa_as_abs = np.zeros(
            (self.N_asizes, self.N_freqs
             ))  # opacities for different grain sizes and wavelengths
        kappa_as_scat = np.zeros(
            (self.N_asizes, self.N_freqs
             ))  # opacities for different grain sizes and wavelengths
        rhoi = self.rho0 * self.f_grain

        #lam = self.lambdas[j]
        #nf0 = self.nf(lam * 1.0e4)
        #kf0 = self.kf(lam * 1.0e4)
        lambdas = self.lambdas
        nfs = self.nf(lambdas * 1.0e4)
        kfs = self.kf(lambdas * 1.0e4)

        get_kappa_as_numba_kernel(len(lambdas), lambdas, nfs, kfs, rhoi,
                                  self.N_asizes, self.a_sizes, kappa_as_abs,
                                  kappa_as_scat, self.f_grain)

        self.kappa_as_abs = kappa_as_abs
        self.kappa_as_scat = kappa_as_scat

    def get_taus_and_kappas(self):

        tau = np.zeros(self.N_freqs)
        tau_abs = np.zeros(self.N_freqs)
        tau_scat = np.zeros(self.N_freqs)
        kappa_abs = np.zeros(self.N_freqs)
        kappa_scat = np.zeros(self.N_freqs)
        omega_nu = np.zeros(self.N_freqs)
        epsilon_nu = np.zeros(self.N_freqs)

        ilogamax = len(self.Sigma_as)

        if self.GoInterp:
            ilogamax = self.ilogamax

        # print("ilogamax", ilogamax, "N_freqs", self.N_freqs)
        for ifreq in range(self.N_freqs):
            tau_abs[ifreq] = np.sum(self.Sigma_as[:ilogamax] *
                                    (self.kappa_as_abs[:ilogamax, ifreq]))
            tau_scat[ifreq] = np.sum(self.Sigma_as[:ilogamax] *
                                     (self.kappa_as_scat[:ilogamax, ifreq]))
            tau[ifreq] = tau_scat[ifreq] + tau_abs[ifreq]
            kappa_abs[ifreq] = tau_abs[ifreq] / self.Sigma_g
            kappa_scat[ifreq] = tau_scat[ifreq] / self.Sigma_g

            omega_nu[ifreq] = tau_scat[ifreq] / (tau_scat[ifreq] +
                                                 tau_abs[ifreq])
            epsilon_nu[ifreq] = 1.0 - omega_nu[ifreq]

        self.tau = tau
        self.tau_abs = tau_abs
        self.tau_scat = tau_scat
        self.kappa_abs = kappa_abs
        self.kappa_scat = kappa_scat
        self.omega_nu = omega_nu
        self.epsilon_nu = epsilon_nu

    def get_Inus(self):
        Inus = np.zeros(self.N_freqs)

        #for ifreq in range(self.N_freqs):
        #    Inus[ifreq] = Bnu_Jy(self.nus[ifreq],
        #                         self.Tdust) * Inu_Bnu_unifslab_direct(
        #                             self.tau[ifreq], self.epsilon_nu[ifreq])
        Inus = Bnu_Jy(self.nus, self.Tdust) * Inu_Bnu_unifslab_direct(
            self.tau, self.epsilon_nu)

        self.Inus = Inus

    def calcul(self):
        if self.ExecTimeReport:
            time_00 = time()
        self.prep()
        if self.ExecTimeReport:
            time_0 = time()
            print("time for prep::", time_0 - time_00, " s")
        self.get_Sigma_as()
        if self.ExecTimeReport:
            time_1 = time()
            print("time for get_Sigma_as::", time_1 - time_0, " s")
        if self.GoInterp:
            self.get_kappa_as_interp()  # 
            #print("GoInterp self.ilogamax", self.ilogamax)
        elif self.GoNumba:
            self.get_kappa_as_numba()
        else:
            self.get_kappa_as()  # 
        if self.ExecTimeReport:
            time_2 = time()
            print("time for get_kappa_as :", time_2 - time_1, " s")
        self.get_taus_and_kappas()
        if self.ExecTimeReport:
            time_3 = time()
            print("time for get_taus_and_kappas :", time_3 - time_2, " s")
        self.get_Inus()
        if self.ExecTimeReport:
            time_4 = time()
            print("time for get_Inus :", time_4 - time_3, " s")

    def get_Plot(self):
        # omega_beam = np.pi * (0.05 * np.pi / (180. * 3600.))**2 # Baobab's brick
        omega_beam = (np.pi /
                      (4. * np.log(2))) * (0.040 * np.pi /
                                           (180. * 3600.))**2  # C10 B3 beam
        BB = Bnu_Jy(self.nus, self.Tdust)
        #overplots=[ [self.nus,omega_beam*BB,'BB']]
        overplots = [[self.nus, omega_beam * BB, 'BB']]
        print("minimum BB/Inu:", min(BB / self.Inus))
        print("maximum tau:", max(self.tau))
        print("minimum tau:", min(self.tau))
        #Plot_Inu(self.nus, self.Inus * omega_beam, overplots=overplots, outputdir=self.outputdir)
        Plot_Inu(self.nus,
                 omega_beam * self.Inus,
                 overplots=overplots,
                 outputdir=self.outputdir)
