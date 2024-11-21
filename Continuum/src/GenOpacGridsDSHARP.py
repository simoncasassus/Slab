import sys
import os
import re
import numpy as np
import matplotlib
from astropy.io import fits

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
from time import time
#from astropy import constants as const
from astropy.constants import astropyconst20 as const
import dsharp_opac as opacity

# matplotlib.use('Agg')
HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/'
sys.path.append(include_path)
import Slab.Continuum.src.AModelSED as AModelSED

c_MKS = const.c.value  # m/s


def gengrid(freqs, ZSetup, filetag=''):

    nfreqs = len(freqs)

    d = np.load(opacity.get_datafile('default_opacities_smooth.npz'))
    #d = np.load(opacity.get_datafile('default_opacities.npz'))
    a = d['a']
    lam = d['lam']
    k_abs = d['k_abs']
    k_sca = d['k_sca']
    gsca = d['g']
    #print("a",a)
    #print(a[1]-a[0])
    #print(a[100]-a[99])
    #print(a[-1]-a[-2])
    #print(np.log10(a[1])-np.log10(a[0]))
    #print(np.log10(a[100])-np.log10(a[99]))
    #print(np.log10(a[-1])-np.log10(a[-2]))

    k_sca_eff = (1 - gsca) * k_sca

    lam_avg = const.c.cgs.value / freqs  # lambdas in cm

    Ngrid_amax = len(a)
    smallestamax = np.min(a)  #cm
    largestamax = np.max(a)  # cm

    print("largestamax", largestamax)
    print("Ngrid_amax", Ngrid_amax)
    amaxs = a.copy()

    Ngrid_qdustexpo = 500
    qdustexpomin = -3.99  #cm
    qdustexpomax = -1.5  # cm
    qdustexpos = (qdustexpomax - qdustexpomin) * (
        np.arange(Ngrid_qdustexpo) / Ngrid_qdustexpo) + qdustexpomin

    kappa_abs_grid = np.zeros((nfreqs, Ngrid_amax, Ngrid_qdustexpo))
    kappa_scat_grid = np.zeros((nfreqs, Ngrid_amax, Ngrid_qdustexpo))

    for iqdustexpo, aqdustexpo in enumerate(qdustexpos):
        #print("iqdustexpo", iqdustexpo, " < ", Ngrid_qdustexpo)

        q = -1 * aqdustexpo  #  [3.5]

        #lam_avg = [0.1, 0.3]  #DEV
        #lam_avg = [0.3, 0.1]  #DEV
        #q = 3.5  # DEV
        #print("lam_avg", lam_avg)
        res_eff = opacity.size_average_opacity(lam_avg,
                                               a,
                                               lam,
                                               k_abs,
                                               k_sca_eff,
                                               q=q,
                                               plot=False)

        #i_lam = 1  # DEV
        #print("res['ka'][i_lam]", res_eff['ka'][i_lam])  # DEV
        #sys.exit()

        kappa_abs_amaxs = res_eff['ka'] / 100
        kappa_scat_amaxs = res_eff['ks'] / 100

        kappa_abs_grid[:, :, iqdustexpo] = kappa_abs_amaxs
        kappa_scat_grid[:, :, iqdustexpo] = kappa_scat_amaxs

    deltaqs = (qdustexpomax - qdustexpomin) / Ngrid_qdustexpo
    deltalogamax = (np.log10(largestamax) -
                    np.log10(smallestamax)) * (1 / Ngrid_amax)
    hdulabs = fits.HDUList()
    hdulscat = fits.HDUList()
    for ifreq, afreq in enumerate(freqs):
        ahdu_abs = fits.PrimaryHDU(kappa_abs_grid[ifreq, :, :])
        ahdu_scat = fits.PrimaryHDU(kappa_scat_grid[ifreq, :, :])
        hdr = ahdu_abs.header
        hdr['freq'] = afreq
        hdr['CDELT2'] = deltalogamax
        hdr['CRVAL2'] = np.log10(smallestamax)
        hdr['CRPIX2'] = 1
        hdr['CTYPE2'] = 'log10(amax)'
        hdr['CDELT1'] = deltaqs
        hdr['CRVAL1'] = qdustexpomin
        hdr['CTYPE1'] = 'q_dustexpo'
        hdr['CRPIX1'] = 1
        ahdu_abs.header = hdr
        ahdu_scat.header = hdr
        hdulabs.append(ahdu_abs)
        hdulscat.append(ahdu_scat)

    if (not (os.path.exists(ZSetup.griddir))):
        os.system("mkdir " + ZSetup.griddir)

    hdulabs.writeto(ZSetup.griddir + 'kappa_abs_grid' + filetag + '.fits',
                    overwrite=True)
    hdulscat.writeto(ZSetup.griddir + 'kappa_scat_grid' + filetag + '.fits',
                     overwrite=True)


#ZSetup = AModelSED.Setup(
#    opct_file='opct_mix.txt',
#    #outputdir='./output_dev_imoptim/')
#    #outputdir='./output_dev_optim_goInterp_wrms/')
#    #outputdir='./output_optim_goInterp_wrms_dsharp/')
#    outputdir='./opac_grids_dsharp/')
#
##obsfreqs = np.array([100E9, 150E9, 230E9, 345E9])
#obsfreqs = np.array([100E9, 150E9, 230E9, 345E9, 694E9])
#
##obsfreqs_alphas = np.array(
##    [100E9, 130E9, 150E9, 180E9, 230E9, 260E9, 345E9, 375E9])
#
#obsfreqs_alphas = np.array(
#    [100E9, 130E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])
#
#gengrid(obsfreqs, ZSetup, filetag='')
#gengrid(obsfreqs_alphas, ZSetup, filetag='_4alphas')
#
