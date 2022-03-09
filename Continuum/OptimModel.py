import sys
import os
import re
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
# from time import time,gmtime, strftime

from astropy import constants as const

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


def Plot_Inu(nus, Inus, outputdir):
    plt.figure(figsize=(10, 4))
    plt.plot(nus / 1E9, Inus, label=r'$I_\nu$')
    plt.plot(nus / 1E9, Inus[-1] * (nus / nus[-1])**2, label=r'$\nu^2$')
    plt.ylabel('Jy/sr')
    plt.xlabel(r'$\nu$ / GHz')
    #plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    fileout = outputdir + 'fig_Inu.pdf'
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
            outputdir += '/'
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
            GenFigs=True,
            opct_file='opct_mix.txt',
            VerboseInit=False,
            outputdir='./output_dev/'):

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
            amin=1E-3,  # cm
            amax=1.,  # cm, maximum grain size
            Sigma_g=0.5,  # g/cm2
            gtod_ratio=100.,
            rho0=2.77,  # g/cm3
            N_asizes=40,
            nus=[],
            ######################################################################
            Sigma_d=0,
            a_sizes=[],
            Sigma_a=[],
            lambdas=[],
            N_freqs=0,
            kappa_as_abs=None,
            kappa_as_scat=None,
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

        self.N_freqs = len(nus)
        self.Sigma_d = self.Sigma_g / self.gtod_ratio
        self.a_sizes = np.logspace(np.log10(self.amin), np.log10(self.amax),
                                   self.N_asizes)

        fas = np.zeros(
            self.N_asizes)  # Surface density for different dust sizes
        for i in range(self.N_asizes):
            if i == 0:
                fas[i] = self.a_sizes[i]**(self.q_dustexpo + 4.)
            else:
                fas[i] = self.a_sizes[i]**(self.q_dustexpo +
                                           4.) - self.a_sizes[i - 1]**(
                                               self.q_dustexpo + 4.)

        self.Sigma_a = fas * self.Sigma_d / np.sum(fas)

        self.lambdas = 1E2 * c_MKS / self.nus  # wavelengths in cm

    def get_kappa_as(self):
        kappa_as_abs = np.zeros(
            (self.N_asizes, self.N_freqs
             ))  # opacities for different grain sizes and wavelengths
        kappa_as_scat = np.zeros(
            (self.N_asizes, self.N_freqs
             ))  # opacities for different grain sizes and wavelengths

        for i_asize, a_asize in enumerate(self.a_sizes):
            rhoi = self.rho0 * self.f_grain
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

    def get_taus_and_kappas(self):

        tau = np.zeros(self.N_freqs)
        tau_abs = np.zeros(self.N_freqs)
        tau_scat = np.zeros(self.N_freqs)
        kappa_abs = np.zeros(self.N_freqs)
        kappa_scat = np.zeros(self.N_freqs)
        omega_nu = np.zeros(self.N_freqs)
        epsilon_nu = np.zeros(self.N_freqs)

        for ifreq in range(self.N_freqs):
            tau_abs[ifreq] = np.sum(self.Sigma_a[:] *
                                    (self.kappa_as_abs[:, ifreq]))
            tau_scat[ifreq] = np.sum(self.Sigma_a[:] *
                                     (self.kappa_as_scat[:, ifreq]))
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
        for ifreq in range(self.N_freqs):
            Inus[ifreq] = Bnu_Jy(self.nus[ifreq],
                                 self.Tdust) * Inu_Bnu_unifslab_direct(
                                     self.tau[ifreq], self.epsilon_nu[ifreq])
        #Inus = Bnu_Jy(self.nus, self.Tdust) * Inu_Bnu_unifslab_direct(
        #    self.tau, self.epsilon_nu)

        self.Inus = Inus

    def get_Plot(self):
        omega_beam = (0.1 * np.pi / (180. * 3600.))**2
        Plot_Inu(self.nus, self.Inus * omega_beam, self.outputdir)
        #Plot_Inu(self.nus, self.Inus, self.outputdir)
