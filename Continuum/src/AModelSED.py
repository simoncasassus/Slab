import sys
import os
import re
import numpy as np
import math
from copy import deepcopy
import matplotlib
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
from time import time
#from astropy import constants as const
from astropy.constants import astropyconst20 as const
from pprint import pprint
import dsharp_opac as opacity

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
    plt.close()

def Plot_Inu(nus,
             Inus,
             overplots=[],
             Nu2Norm=False,
             outputdir='',
             ExtraLabel='',
             fileout='fig_Inu.pdf'):
    plt.figure(figsize=(10, 4))

    Inorm = (nus / 100E9)**2

    for iover, aover in enumerate(overplots):
        anus = aover[0]
        aInus = aover[1]
        if Nu2Norm:
            aInus /= Inorm
        label = aover[2]
        plt.plot(anus / 1E9, aInus, label=label)

    label = r'$I_\nu$'
    if Nu2Norm:
        Inus /= Inorm
        label = r'$I_\nu ~/ ~ (\nu / 100 {\rm GHz})^2$'
    plt.plot(nus / 1E9, Inus, label=label + '\n' + ExtraLabel)
    #plt.plot(nus / 1E9, Inus[-1] * (nus / nus[-1])**2, label=r'$\nu^2$')
    #plt.ylabel('Jy/sr')
    plt.ylabel('Jy/beam')
    #plt.ylabel('Jy/brick')
    plt.xlabel(r'$\nu$ / GHz')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    fileout = outputdir + fileout
    print(fileout)
    plt.savefig(fileout, bbox_inches='tight')


def Plot_kappa(nus, kappa_abs, kappa_scat, outputdir='', ExtraLabel='', fileout='fig_kappas.pdf'):
    
    plt.figure(figsize=(10, 4))

    label_a = r'$\kappa^{\rm abs}_\nu$'
    label_s = r'$\kappa^{\rm scat}_\nu$'
    plt.plot(nus / 1E9, kappa_abs, label=label_a + '\n' + ExtraLabel)
    plt.plot(nus / 1E9, kappa_scat, label=label_s + '\n' + ExtraLabel)
    plt.ylabel(r'${\rm cm}^2 {\rm g}^{-1}$')
    plt.xlabel(r'$\nu$ / GHz')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    fileout = outputdir + fileout
    print(fileout)
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
        #Qabs = 1./(1./Qabs2 + 1./Qabs3)
        
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
        #Qabs = 1./(1./Qabs2 + 1./Qabs3)

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
        #Qscat = 1./(1./Qscat2 + 1./Qscat3)
    #print x, lam
    kappa0 = 3. / (4. * a * rho)
    return kappa0 * Qscat


@jit(nopython=True)
def bilin(i1, j1, i2, j2, dx, dy, cube, ifreq):
    #  Bilinear interpolation
    v11 = cube[ifreq, i1, j1]
    v12 = cube[ifreq, i1, j2]
    v21 = cube[ifreq, i2, j1]
    v22 = cube[ifreq, i2, j2]
    value = (1. - dx) * (1. - dy) * v11 + (1. - dx) * dy * v12 + dx * (
        1. - dy) * v21 + dx * dy * v22
    return value


@jit(nopython=True)
def get_taus_and_kappas_interp2D_kernel(Sigma_g, N_freqs, i1, i2, j1, j2,
                                        dlogamax, dq, kappa_abs_nus_grid,
                                        kappa_scat_nus_grid, kappa_abs,
                                        kappa_scat, tau_abs, tau_scat, tau,
                                        omega_nu, epsilon_nu):

    for ifreq in range(N_freqs):
        #hdukappa_abs = kappa_abs_hdulgrid[ifreq]
        #kappa_abs2D = hdukappa_abs.data
        dum = bilin(i1, j1, i2, j2, dlogamax, dq, kappa_abs_nus_grid, ifreq)
        kappa_abs[ifreq] = dum

        #hdukappa_scat = kappa_scat_hdulgrid[ifreq]
        #kappa_scat2D = hdukappa_scat.data
        #kappa_scat2D = kappa_scat_nus_grid[ifreq, :, :]
        #kappa_scat[ifreq] = bilin(i1, j1, i2, j2, dlogamax, dq, kappa_scat_nus_grid, ifreq)
        kappa_scat[ifreq] = bilin(i1, j1, i2, j2, dlogamax, dq,
                                  kappa_scat_nus_grid, ifreq)

        tau_abs[ifreq] = kappa_abs[ifreq] * Sigma_g
        tau_scat[ifreq] = kappa_scat[ifreq] * Sigma_g
        tau[ifreq] = tau_scat[ifreq] + tau_abs[ifreq]
        omega_nu[ifreq] = tau_scat[ifreq] / (tau_scat[ifreq] + tau_abs[ifreq])
        epsilon_nu[ifreq] = 1.0 - omega_nu[ifreq]


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


@jit(nopython=True)
def get_taus_and_kappas_numba_kernel(Sigma_g, N_freqs, ilogamax, Sigma_as,
                                     kappa_as_abs, kappa_as_scat, tau, tau_abs,
                                     tau_scat, kappa_abs, kappa_scat, omega_nu,
                                     epsilon_nu):

    for ifreq in range(N_freqs):
        tau_abs[ifreq] = 0.
        tau_scat[ifreq] = 0.

        for isize in range(ilogamax):
            tau_abs[ifreq] += Sigma_as[isize] * (kappa_as_abs[isize, ifreq])
            tau_scat[ifreq] += Sigma_as[isize] * (kappa_as_scat[isize, ifreq])
            #if (tau_abs[ifreq] < 0):
            #    print("Sigma_as[isize]", Sigma_as[isize],
            #          "kappa_as_abs[isize, ifreq]", kappa_as_abs[isize, ifreq])

        tau[ifreq] = tau_scat[ifreq] + tau_abs[ifreq]
        kappa_abs[ifreq] = tau_abs[ifreq] / Sigma_g
        kappa_scat[ifreq] = tau_scat[ifreq] / Sigma_g

        omega_nu[ifreq] = tau_scat[ifreq] / (tau_scat[ifreq] + tau_abs[ifreq])
        epsilon_nu[ifreq] = 1.0 - omega_nu[ifreq]


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
        #Qscat = 1./(1./Qscat2 + 1./Qscat3)
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
            PrintChi2s=True,
            ClearOutputDir=False,
            nf=None,
            kf=None,
            #log_a_sizes_4nearneighbor1D=None,
            GoNearNeighbor1D=False,
            GoInterp=False,
            amax_4grid=10,  #cm
            GenFigs=False,
            opct_file='opct_mix.txt',
            VerboseInit=False,
            use_dsharp_opac=True,  # if true, computes dsharp grid, if false, and GoInterp is also false,  uses Kataoka+ 2014
            outputdir='./output_dev/',
            griddir='./opac_grids/',
            ######################################################################
            kappa_as_abs_4nearneighbor1D=None,
            kappa_as_scat_4nearneighbor1D=None,
            kappa_as_abs=None,
            kappa_as_scat=None):

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
            N_asizes=40,  #if ZSetup.GoNearNeighbor1D  then N_asizes should be set to the grid
            nus=[],
            ExecTimeReport=False,
            GoNumba=True,
            Verbose=True,
            omega_beam_4plots=(np.pi /
                                 (4. * np.log(2))) * (0.040 * np.pi /
                                                      (180. * 3600.))**2,  # C10 B3 beam
            ######################################################################
            Inus=[],
            Sigma_d=0,
            Sigma_g_0=100.,  # optically thin Beckwith column
            Tdust_0=30.,  # optically thin Beckwith column
            Tdust_high=1000.,  # optically thin Beckwith column
            Tdust_low=10.,  # optically thin Beckwith column
            a_sizes=[],
            Sigma_as=[],
            lambdas=[],
            N_freqs=0,
            nfs=None,
            kfs=None,
            kappa_as_abs=None,
            kappa_as_scat=None,
            ilogamax=None,
            tau_abs=None,
            tau_scat=None,
            tau=None,
            kappa_abs=None,
            kappa_scat=None,
            gridfiletag='',
            kappa_abs_hdulgrid=None,
            kappa_scat_hdulgrid=None,
            kappa_abs_nus_grid=None,
            kappa_scat_nus_grid=None,
            logamaxgrid=None,
            Nlogamaxgrid=None,
            logamaxgrid_smallest=None,
            logamaxgrid_delta=None,
            qgrid_smallest=None,
            qgrid_delta=None,
            qgrid=None,
            Nqgrid=None):

        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            if ASetup.VerboseInit:
                print("MSED> setting ", a_attribute, " to ",
                      initlocals[a_attribute])
            setattr(self, a_attribute, initlocals[a_attribute])

        self.__dict__.update(ASetup.__dict__)

        #if self.nfs is None:
        #    print("initialize nfs kfs")
        #    lambdas = self.lambdas
        #    self.nfs = self.nf(lambdas * 1.0e4)
        #    self.kfs = self.kf(lambdas * 1.0e4)

    def prep(self, ForcePrep=False):
        self.N_freqs = len(self.nus)
        self.lambdas = 1E2 * c_MKS / self.nus  # wavelengths in cm
        self.Sigma_d = self.Sigma_g / self.gtod_ratio
        #self.a_sizes = np.logspace(np.log10(self.amin), np.log10(self.amax),
        #                           self.N_asizes)

        if self.GoInterp:
            #print("ForcePrep",ForcePrep,"self.kappa_abs_nus_grid",type(self.kappa_abs_nus_grid))
            if (self.kappa_abs_nus_grid is None) or ForcePrep:
                #print("init 2D interp",type(self.kappa_abs_nus_grid),"ForcePrep",ForcePrep)
                filetag = self.gridfiletag
                if os.path.exists(self.griddir + 'kappa_abs_grid' + filetag +
                                  '.fits'):
                    if self.Verbose:
                        print("loading gridded kappas "+filetag+" from directory ",  self.griddir)
                    #print("filetag", filetag)
                    self.kappa_abs_hdulgrid = fits.open(self.griddir +
                                                        'kappa_abs_grid' +
                                                        filetag + '.fits')
                    self.kappa_scat_hdulgrid = fits.open(self.griddir +
                                                         'kappa_scat_grid' +
                                                         filetag + '.fits')
                    hdr = self.kappa_abs_hdulgrid[0].header
                    self.logamaxgrid = hdr['CRVAL2'] + hdr['CDELT2'] * (
                        np.arange(hdr['NAXIS2']) - (hdr['CRPIX2'] - 1))
                    self.logamaxgrid_smallest = hdr['CRVAL2']
                    self.logamaxgrid_delta = hdr['CDELT2']
                    self.Nlogamaxgrid = hdr['NAXIS2']
                    self.qgrid = hdr['CRVAL1'] + hdr['CDELT1'] * (
                        np.arange(hdr['NAXIS1']) - (hdr['CRPIX1'] - 1))
                    self.qgrid_smallest = hdr['CRVAL1']
                    self.qgrid_delta = hdr['CDELT1']
                    self.Nqgrid = hdr['NAXIS1']

                    gridshape = list(self.kappa_abs_hdulgrid[0].data.shape)
                    gridshape.insert(0, self.N_freqs)
                    #xprint("gridshape", gridshape)
                    kappa_abs_nus_grid = np.zeros(gridshape)
                    kappa_scat_nus_grid = np.zeros(gridshape)
                    for ifreq in range(self.N_freqs):
                        #print("ifreq", ifreq, self.N_freqs,
                        #len(self.kappa_abs_hdulgrid))
                        kappa_abs_nus_grid[
                            ifreq, :, :] = self.kappa_abs_hdulgrid[ifreq].data
                        kappa_scat_nus_grid[
                            ifreq, :, :] = self.kappa_scat_hdulgrid[ifreq].data
                    self.kappa_abs_nus_grid = kappa_abs_nus_grid
                    self.kappa_scat_nus_grid = kappa_scat_nus_grid

                else:
                    sys.exit("compute grids first")
        else:
            if (self.nfs is None) or ForcePrep:
                if self.Verbose:
                    print("initialize nfs kfs")
                lambdas = self.lambdas
                self.nfs = self.nf(lambdas * 1.0e4)
                self.kfs = self.kf(lambdas * 1.0e4)

            if not self.GoNearNeighbor1D:
                loga_sizes = (np.log10(self.amax) - np.log10(self.amin)) * (
                    np.arange(self.N_asizes) / self.N_asizes) + np.log10(
                        self.amin)
                self.a_sizes = 10**(loga_sizes)

            if (self.kappa_as_abs is None) or ForcePrep:
                if self.Verbose:
                    print("initializae kappa_as arrays")
                self.kappa_as_abs = np.zeros(
                    (self.N_asizes, self.N_freqs
                     ))  # opacities for different grain sizes and wavelengths
                self.kappa_as_scat = np.zeros(
                    (self.N_asizes, self.N_freqs
                     ))  # opacities for different grain sizes and wavelengths

        if self.GoNearNeighbor1D:
            if (self.kappa_as_abs_4nearneighbor1D is None) or ForcePrep:
                print("computing opacity grid using Kataoka formulae")
                amax = self.amax
                self.amax = self.amax_4grid
                loga_sizes = (np.log10(self.amax) - np.log10(self.amin)) * (
                    np.arange(self.N_asizes) / self.N_asizes) + np.log10(
                        self.amin)
                self.a_sizes = 10**(loga_sizes)
                self.get_kappa_as_numba()
                self.kappa_as_abs_4nearneighbor1D = self.kappa_as_abs
                self.kappa_as_scat_4nearneighbor1D = self.kappa_as_scat
                self.kappa_as_abs_4nearneighbor1D = self.kappa_as_abs
                self.kappa_as_scat_4nearneighbor1D = self.kappa_as_scat
                self.amax = amax

    #def copy(self, AnotherSED):
    #    #self.__dict__.update(AnotherSED.__dict__)
    #    attributes_source = AnotherSED.__dict__
    #    for anattribute in attributes_source.keys():
    #        value_source = getattr(AnotherSED, anattribute)
    #        attributes_source[anattribute] = deepcopy(value_source)
    #        #if isinstance(value_source, np.ndarray):
    #        #    #print("found ndarray ", anattribute)
    #        #    attributes_source[anattribute] = value_source.copy()
    #    self.__dict__.update(attributes_source)

    def get_Sigma_as(self):

        #fas = np.zeros(
        #    self.N_asizes)  # Surface density for different dust sizes
        #for i in range(self.N_asizes):
        #    if i == 0:
        #        fas[i] = self.a_sizes[i]**(self.q_dustexpo + 4.)
        #    else:
        #        fas[i] = self.a_sizes[i]**(self.q_dustexpo +
        #                                   4.) - self.a_sizes[i - 1]**(
        #                                       self.q_dustexpo + 4.)

        ilogamax = len(self.a_sizes)
        #print("ilogamax in get_Sigma_as",ilogamax)
        if self.GoNearNeighbor1D:
            ilogamax = self.ilogamax
        fas2 = self.a_sizes[0:ilogamax]**(self.q_dustexpo + 4)
        fas1 = np.roll(fas2, 1)
        fas = fas2 - fas1
        fas[0] = self.a_sizes[0]**(self.q_dustexpo + 4.)

        self.Sigma_as = fas * self.Sigma_d / np.sum(fas)

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

    def get_kappa_as_nearneighbor1D(self):

        logamax = np.log10(self.amax)
        dlogas = (np.log10(self.amax_4grid) -
                  np.log10(self.amin)) / self.N_asizes
        ilogamax = int((logamax - np.log10(self.amin)) / dlogas)

        self.kappa_as_abs = self.kappa_as_abs_4nearneighbor1D
        self.kappa_as_scat = self.kappa_as_scat_4nearneighbor1D
        self.ilogamax = ilogamax + 1

    def get_kappa_as_numba(self):
        #kappa_as_abs = np.zeros(
        #    (self.N_asizes, self.N_freqs
        #     ))  # opacities for different grain sizes and wavelengths
        #kappa_as_scat = np.zeros(
        #    (self.N_asizes, self.N_freqs
        #     ))  # opacities for different grain sizes and wavelengths

        rhoi = self.rho0 * self.f_grain

        #lam = self.lambdas[j]
        #nf0 = self.nf(lam * 1.0e4)
        #kf0 = self.kf(lam * 1.0e4)

        get_kappa_as_numba_kernel(len(self.lambdas), self.lambdas, self.nfs,
                                  self.kfs, rhoi, self.N_asizes, self.a_sizes,
                                  self.kappa_as_abs, self.kappa_as_scat,
                                  self.f_grain)

        #self.kappa_as_abs = kappa_as_abs
        #self.kappa_as_scat = kappa_as_scat

    def get_taus_and_kappas_interp(self):
        tau = np.zeros(self.N_freqs)
        tau_abs = np.zeros(self.N_freqs)
        tau_scat = np.zeros(self.N_freqs)
        kappa_abs = np.zeros(self.N_freqs)
        kappa_scat = np.zeros(self.N_freqs)
        omega_nu = np.zeros(self.N_freqs)
        epsilon_nu = np.zeros(self.N_freqs)

        kappa_abs_hdulgrid = self.kappa_abs_hdulgrid
        kappa_scat_hdulgrid = self.kappa_scat_hdulgrid

        Sigma_g = self.Sigma_g

        logamax = np.log10(self.amax)
        q_dustexpo = self.q_dustexpo
        logamaxgrid = self.logamaxgrid

        dlogamax = self.logamaxgrid_delta
        i1 = math.floor((logamax - self.logamaxgrid_smallest) / dlogamax)
        i2 = i1 + 1
        if i1 < 0:
            i1 = 0
            i2 = 1
        if i2 >= self.Nlogamaxgrid:
            i1 = int(self.Nlogamaxgrid - 2)
            i2 = int(self.Nlogamaxgrid - 1)

        #if i1 < 0 or i2 >= self.Nlogamaxgrid:
        #    print("logamax",logamax,"beyond bounds")
        #    print("i1 ", i1, "i2", i2, "Nlogamaxgrid", self.Nlogamaxgrid)
        #    sys.exit("extrapolate logamax?")

        dq = self.qgrid_delta
        j1 = math.floor((q_dustexpo - self.qgrid_smallest) / dq)
        j2 = j1 + 1
        if (j1 < 0):
            j1 = 0
            j2 = 1
        if (j2 >= self.Nqgrid):
            j1 = int(self.Nqgrid - 2)
            j2 = int(self.Nqgrid - 1)

        #if (j1 < 0) or (j2 >= self.Nqgrid):
        #    print("q_dustexpo", q_dustexpo, "beyond bounds")
        #    print("j1 ", j1, "j2", j2, "Nqgrid", self.Nqgrid)
        #    sys.exit("extrapolate q_dustexpo?")

        N_freqs = self.N_freqs
        kappa_abs_nus_grid = self.kappa_abs_nus_grid
        kappa_scat_nus_grid = self.kappa_scat_nus_grid
        get_taus_and_kappas_interp2D_kernel(Sigma_g, N_freqs, i1, i2, j1, j2,
                                            dlogamax, dq, kappa_abs_nus_grid,
                                            kappa_scat_nus_grid, kappa_abs,
                                            kappa_scat, tau_abs, tau_scat, tau,
                                            omega_nu, epsilon_nu)

        self.tau = tau
        self.tau_abs = tau_abs
        self.tau_scat = tau_scat
        self.kappa_abs = kappa_abs
        self.kappa_scat = kappa_scat
        self.omega_nu = omega_nu
        self.epsilon_nu = epsilon_nu

    def get_taus_and_kappas(self):

        tau = np.zeros(self.N_freqs)
        tau_abs = np.zeros(self.N_freqs)
        tau_scat = np.zeros(self.N_freqs)
        kappa_abs = np.zeros(self.N_freqs)
        kappa_scat = np.zeros(self.N_freqs)
        omega_nu = np.zeros(self.N_freqs)
        epsilon_nu = np.zeros(self.N_freqs)

        #ilogamax = len(self.Sigma_as)
        ilogamax = len(self.a_sizes)

        if self.GoNearNeighbor1D:
            ilogamax = self.ilogamax

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

    def get_taus_and_kappas_numba(self):

        tau = np.zeros(self.N_freqs)
        tau_abs = np.zeros(self.N_freqs)
        tau_scat = np.zeros(self.N_freqs)
        kappa_abs = np.zeros(self.N_freqs)
        kappa_scat = np.zeros(self.N_freqs)
        omega_nu = np.zeros(self.N_freqs)
        epsilon_nu = np.zeros(self.N_freqs)

        #ilogamax = len(self.Sigma_as)
        ilogamax = len(self.a_sizes)

        if self.GoNearNeighbor1D:
            ilogamax = self.ilogamax

        get_taus_and_kappas_numba_kernel(self.Sigma_g, self.N_freqs, ilogamax,
                                         self.Sigma_as, self.kappa_as_abs,
                                         self.kappa_as_scat, tau, tau_abs,
                                         tau_scat, kappa_abs, kappa_scat,
                                         omega_nu, epsilon_nu)

        #print("tau_abs", tau_abs)

        self.tau = tau
        self.tau_abs = tau_abs
        self.tau_scat = tau_scat
        self.kappa_abs = kappa_abs
        self.kappa_scat = kappa_scat
        self.omega_nu = omega_nu
        self.epsilon_nu = epsilon_nu

    def get_taus_and_kappas_dsharp(self):

        if self.Verbose:
            print("computing kappas using dsharp_opac")
        d = np.load(opacity.get_datafile('default_opacities_smooth.npz'))
        #d = np.load(opacity.get_datafile('default_opacities.npz'))
        a = d['a']
        lam = d['lam']
        k_abs = d['k_abs']
        k_sca = d['k_sca']
        gsca = d['g']

        k_sca_eff = (1 - gsca) * k_sca
        #eps_nu = k_abs / (k_abs + k_sca_eff)

        # lam_avg = [0.1, 0.3]
        lam_avg = const.c.cgs.value / self.nus  # lambdas in cm
        q = -1 * self.q_dustexpo  #  [3.5]

        #check if lam_avg is sorted:
        is_sorted = lambda a: np.all(a[:-1] <= a[1:])
        res = is_sorted(lam_avg)
        if not res:
            print("lam_avg",lam_avg)
            print("observed frequencies must be in descending order for dsharp_opac")
            sys.exit("observed frequencies must be in descending order for dsharp_opac")
        
        
        #res = [
        #    opacity.size_average_opacity(lam_avg,
        #                                 a,
        #                                 lam,
        #                                 k_abs,
        #                                 k_sca,
        #                                 q=_q,
        #                                 plot=False) for _q in q
        #]

        res_eff = opacity.size_average_opacity(lam_avg,
                                               a,
                                               lam,
                                               k_abs,
                                               k_sca_eff,
                                               q=q,
                                               plot=False)

        kappa_abs_amaxs = res_eff['ka'] / 100
        kappa_scat_amaxs = res_eff['ks'] / 100

        kappa_abs = np.zeros(self.N_freqs)
        kappa_scat = np.zeros(self.N_freqs)

        for ifreq in range(self.N_freqs):
            kappa_abs[ifreq] = np.interp(self.amax, a,
                                         kappa_abs_amaxs[ifreq, :])
            kappa_scat[ifreq] = np.interp(self.amax, a,
                                          kappa_scat_amaxs[ifreq, :])

        tau_abs = self.Sigma_g * kappa_abs
        tau_scat = self.Sigma_g * kappa_scat
        tau = tau_abs + tau_scat

        omega_nu = tau_scat / (tau_scat + tau_abs)
        epsilon_nu = 1.0 - omega_nu

        #pprint(res_eff)
        #print("a sizes dsharp ", len(a))
        #print("kappa_abs.shape", kappa_abs.shape)
        ##print("kappa_abs[10,0]",kappa_abs[10,0])
        ##print("kappa_abs[10,-1]",kappa_abs[10,-1])
        ##print("kappa_abs[10,100]",kappa_abs[10,100])
        #
        #print("self.N_freqs", self.N_freqs)
        #print("self.nus.shape", self.nus.shape)
        #print("tau.shape", tau.shape)
        #print("omega_nu.shape", omega_nu.shape)

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

    def calcul(self, ForcePrep=False):
        if self.ExecTimeReport:
            time_00 = time()
        self.prep(ForcePrep=ForcePrep)
        if self.ExecTimeReport:
            time_0 = time()
            print("time for prep::", time_0 - time_00, " s")
        if self.GoInterp:
            self.get_taus_and_kappas_interp()
            if self.ExecTimeReport:
                time_3 = time()
                print("time for get_taus_and_kappas :", time_3 - time_0, " s")
        elif self.use_dsharp_opac:
            self.get_taus_and_kappas_dsharp()
            if self.ExecTimeReport:
                time_3 = time()
                print("time for get_taus_and_kappas_dsharp :", time_3 - time_0,
                      " s")
        else:
            self.get_Sigma_as()
            if self.ExecTimeReport:
                time_1 = time()
                print("time for get_Sigma_as::", time_1 - time_0, " s")
            if self.GoNearNeighbor1D:
                self.get_kappa_as_nearneighbor1D()  #
                #print("GoNearNeighbor1D self.ilogamax", self.ilogamax)
            elif self.GoNumba:
                self.get_kappa_as_numba()
            else:
                self.get_kappa_as()  #
            if self.ExecTimeReport:
                time_2 = time()
                print("time for get_kappa_as :", time_2 - time_1, " s")
            if self.GoNumba:
                self.get_taus_and_kappas_numba()
            else:
                self.get_taus_and_kappas()
            if self.ExecTimeReport:
                time_3 = time()
                print("time for get_taus_and_kappas :", time_3 - time_2, " s")
        self.get_Inus()
        if self.ExecTimeReport:
            time_4 = time()
            print("time for get_Inus :", time_4 - time_3, " s")

    def get_Plot(self, drawBB=True, Nu2Norm=False, ExtraLabel=''):
        # omega_beam = np.pi * (0.05 * np.pi / (180. * 3600.))**2 # Baobab's brick
        omega_beam = self.omega_beam_4plots
        #overplots=[ [self.nus,omega_beam*BB,'BB']]
        overplots = []
        if drawBB:
            BB = Bnu_Jy(self.nus, self.Tdust)
            overplots = [[self.nus, omega_beam * BB, 'BB']]
            print("minimum BB/Inu:", min(BB / self.Inus))
        print("maximum tau:", max(self.tau))
        print("minimum tau:", min(self.tau))
        #Plot_Inu(self.nus, self.Inus * omega_beam, overplots=overplots, outputdir=self.outputdir)
        Plot_Inu(self.nus,
                 omega_beam * self.Inus,
                 overplots=overplots,
                 Nu2Norm=Nu2Norm,
                 ExtraLabel=ExtraLabel,
                 outputdir=self.outputdir)

    def get_kappa_Plot(self, drawBB=True, Nu2Norm=False, ExtraLabel=''):
        Plot_kappa(self.nus,
                   self.kappa_abs,
                   self.kappa_scat,
                   ExtraLabel=ExtraLabel,
                   outputdir=self.outputdir)

