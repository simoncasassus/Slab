import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
# from time import time,gmtime, strftime
import sys

from astropy import constants as const

c_MKS = const.c.value  # m/s


def Plot_Opct(Opct):
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
    plt.savefig(fileout, bbox_inches='tight')


def Plot_Inu(nus, Inus):
    plt.figure(figsize=(10, 4))
    plt.plot(nus / 1E9, Inus, label=r'$I_\nu$')
    plt.plot(nus / 1E9, Inus[-1] * (nus / nus[-1])**2, label=r'$\nu^2$')
    plt.ylabel('Jy/sr')
    plt.xlabel(r'$\nu$ / GHz')
    #plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    fileout = 'fig_Inu.pdf'
    plt.savefig(fileout, bbox_inches='tight')


def load_Opct(opct_file='opct_mix.txt', GenFigs=False):
    Opct = np.loadtxt(
        opct_file)  # mix of silicates, amorphous carbon and water ice
    if GenFigs:
        Plot_Opct(Opct)
    nf = interp1d(Opct[:, 0], Opct[:, 1], kind='cubic')
    kf = interp1d(Opct[:, 0], Opct[:, 2], kind='cubic')
    return nf, kf


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
def f_kappa(nf, kf, a, lam, rho, f):
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



(nf, kf) = load_Opct(opct_file='opct_mix.txt', GenFigs=True)


Tdust = 30.

q_dustexpo = -3.5
f_grain = 1.  # grain filling factor
amin = 1E-3  # cm
amax = 1.  # cm, maximum grain size
Sigma_g = 0.5  # g/cm2
gtod_ratio = 100.
rho0 = 2.77  # g/cm3

# plot_opacities(nf, kf,amax=amax,rho0=rho0,f_grain=f_grain)

Sigma_d = Sigma_g / gtod_ratio
N_asizes = 40
a_sizes = np.logspace(np.log10(amin), np.log10(amax), N_asizes)

fracs_mass = np.zeros(N_asizes)  # mass fractions

fas = np.zeros(
    N_asizes)  # array to save Surface density for different dust sizes

for i in range(N_asizes):
    if i == 0:
        fas[i] = a_sizes[i]**(q_dustexpo + 4.)
    else:
        fas[i] = a_sizes[i]**(q_dustexpo + 4.) - a_sizes[i - 1]**(q_dustexpo +
                                                                  4.)

Sigma_a = fas * Sigma_d / np.sum(fas)

N_freqs = 100
lognu1 = np.log10(30E9)
lognu2 = np.log10(700E9)
lognus = lognu1 + (np.arange(N_freqs) / N_freqs) * (lognu2 - lognu1)
nus = 10**lognus

lams = 1E2 * c_MKS / nus  # wavelengths in cm

kappas = np.zeros(
    (N_asizes, N_freqs
     ))  # array to save opacities for different grain sizes and wavelengths
kappas_scat = np.zeros(
    (N_asizes, N_freqs
     ))  # array to save opacities for different grain sizes and wavelengths

for i_asize, a_asize in enumerate(a_sizes):
    rhoi = rho0 * f_grain
    for j in range(len(lams)):
        kappas[i_asize, j] = f_kappa(nf, kf, a_asize, lams[j], rhoi, f_grain)
        kappas_scat[i_asize, j] = f_kappa_scat(nf, kf, a_asize, lams[j], rhoi,
                                               f_grain)

tau = np.zeros(N_freqs)
tau_abs = np.zeros(N_freqs)
tau_scat = np.zeros(N_freqs)
Inus = np.zeros(N_freqs)

for ifreq in range(N_freqs):

    tau[ifreq] = np.sum(Sigma_a[:] *
                        (kappas[:, ifreq] + kappas_scat[:, ifreq]))
    tau_abs[ifreq] = np.sum(Sigma_a[:] * (kappas[:, ifreq]))
    tau_scat[ifreq] = np.sum(Sigma_a[:] * (kappas_scat[:, ifreq]))

    omega_nu = tau_scat[ifreq] / (tau_scat[ifreq] + tau_abs[ifreq])
    epsilon_nu = 1.0 - omega_nu

    Inus[ifreq] = Bnu_Jy(nus[ifreq], Tdust) * Inu_Bnu_unifslab_direct(
        tau[ifreq], epsilon_nu)

omega_beam = (0.1 * np.pi / (180. * 3600.))**2

Plot_Inu(nus, Inus * omega_beam)
