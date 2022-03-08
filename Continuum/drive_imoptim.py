import os
import numpy as np
import matplotlib
from astropy.io import fits
from copy import deepcopy

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
# from time import time,gmtime, strftime
import sys

from astropy import constants as const

HOME = os.environ.get('HOME')
include_path = HOME + '/common/python/include'
sys.path.append(include_path)

import PyVtools.Vtools as Vtools
from ImUtils.Resamp import gridding
from ImUtils.Cube2Im import slice0
from Gausssmooth import Gauss_filter

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/Slab/Continuum/'
sys.path.append(include_path)

import AModelSED
import SEDOptim


def load_imagfile(file_data, zoomfactor=1., Debug=False):

    #f0 = fits.open(file)
    hdu = slice0(file_data, ReturnHDUList=True)
    im0 = hdu[0].data
    hdr0 = hdu[0].header

    (ny, nx) = im0.shape
    pixscale = hdr0['CDELT2'] * 3600.
    print("data original pixscale = ", pixscale)

    hdr = deepcopy(hdr0)
    hdr['CDELT1'] *= zoomfactor
    hdr['CDELT2'] *= zoomfactor
    hdr['NAXIS1'] /= zoomfactor
    hdr['NAXIS2'] /= zoomfactor
    nx /= zoomfactor
    ny /= zoomfactor
    i0 = int((nx - 1) / 2)
    j0 = int((ny - 1) / 2)

    hdr['CRPIX1'] = i0 + 1
    hdr['CRPIX2'] = j0 + 1

    f1 = gridding(hdu, hdr, ReturnHDUList=True)
    im = f1[0].data

    pixscale = hdr['CDELT2'] * 3600.

    print("resampled pixscale = ", pixscale)

    ivec = np.arange(0, nx)
    jvec = np.arange(0, ny)
    ii, jj = np.meshgrid(ivec, jvec)

    dxxs = -pixscale * (ii - i0)
    dyys = pixscale * (jj - i0)
    rrs = np.sqrt((dxxs)**2 + (dyys)**2)

    hdr['PIXSCALE'] = pixscale
    hdu[0].data = im
    hdu[0].hdr = hdr

    return rrs, hdu, pixscale


def punchmap(im, hdu, units='', fileout='test.fits'):
    hdr = hdu[0].header
    hdr['BUNIT'] = units
    hdu[0].header = hdr
    hdu[0].data = im
    hdu.writeto(fileout,overwrite=True)


zoomfactor = 8

files_images = [
    './data/I_100.fits',
    './data/I_150.fits',
    './data/I_230.fits',
    './data/I_345.fits',
]

files_specindex = [
    './data/specindec_100.fits',
    './data/specindec_150.fits',
    './data/specindec_230.fits',
    './data/specindec_345.fits',
]

mfreq_imhdus = []
for afile in files_images:
    rrs, hdu, pixscale = load_imagfile(afile, zoomfactor=zoomfactor)
    mfreq_imhdus.append(hdu)

print(len(mfreq_imhdus))
hdu_canvas = mfreq_imhdus[0]

Vtools.View(hdu_canvas)

mfreq_specindexhdus = []
for afile in files_images:
    rrs2, hdu, pixscale = load_imagfile(afile, zoomfactor=zoomfactor)
    mfreq_specindexhdus.append(hdu)

print(len(mfreq_specindexhdus))

outputdir = './output_dev_imoptim/'

ZSetup = AModelSED.Setup(
    filetag='',  # False
    Verbose=True,
    PrintChi2s=True,
    ClearOutputDir=False,
    GenFigs=False,
    opct_file='opct_mix.txt',
    VerboseInit=False,
    outputdir=outputdir)

obsfreqs = np.array([100E9, 150E9, 230E9, 345E9])

fluxcal_accuracy = np.array([0.1, 0.1, 0.1, 0.1])
fluxcal_accuracy = np.array([0.01, 0.01, 0.01, 0.01])

obsfreqs_alphas = np.array(
    [100E9, 115E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])

obsnu1s = np.array([100E9, 150E9, 230E9, 345E9])

obsnu2s = np.array([115E9, 165E9, 245E9, 360E9])

obsfreqs_alphas = np.array(
    [100E9, 130E9, 150E9, 180E9, 230E9, 260E9, 345E9, 375E9])

ZData = SEDOptim.Data()
ZData.nus = obsfreqs
ZData.nu1s_alphas = obsnu1s
ZData.nu2s_alphas = obsnu2s
ZData.nus_alphas = obsfreqs_alphas

ASED = AModelSED.MSED(
    ZSetup,
    Tdust=30.,
    q_dustexpo=-3.5,
    f_grain=1.,  # grain filling factor
    amin=1E-3,  # cm
    amax=1.,  # cm, maximum grain size
    Sigma_g=50.,  # g/cm2
    gtod_ratio=100.,
    rho0=2.77,  # g/cm3
    N_asizes=400,
    GoNumba=True,
    nus=obsfreqs)

ZMerit = SEDOptim.Merit(ExecTimeReport=False)
domain = [
    ['log(Tdust)', np.log10(30.), [0., 3]],
    ['q_dustexpo', -3.5, [-3.99, -2.]],
    #['f_grain', 1., [0., 1.]],
    ['log(amax)', np.log10(1.), [np.log10(1E-3), np.log10(10.)]],  #cm
    ['log(Sigma_g)',
     np.log10(50.), [np.log10(1E-5), np.log10(1E3)]]
]  # g/cm2

nvars = len(domain)
print("nvars: ", nvars)
OptimM = SEDOptim.OptimM(
    RunMCMC=True,
    MCMC_Nit=100,  # 200 MCMC iterations
    nwalkers_pervar=10,  # 10
    burn_in=50,  #100
    n_cores_MCMC=6,
    ChainPlots=False,
    CornerPlots=False,
    Report=False,
    MCMCProgress=False,
    SummaryPlots=False,
    domain=domain)

intraband_accuracy = 0.008

nfreqs = len(mfreq_imhdus)
nspecindexs = len(mfreq_specindexhdus)
im_canvas = hdu_canvas[0].data

imlogTdust = np.zeros(im_canvas.shape)
supimlogTdust = np.zeros(im_canvas.shape)
sdoimlogTdust = np.zeros(im_canvas.shape)
imq_dustexpo = np.zeros(im_canvas.shape)
supimq_dustexpo = np.zeros(im_canvas.shape)
sdoimq_dustexpo = np.zeros(im_canvas.shape)
imlogamax = np.zeros(im_canvas.shape)
supimlogamax = np.zeros(im_canvas.shape)
sdoimlogamax = np.zeros(im_canvas.shape)
imlogSigma_g = np.zeros(im_canvas.shape)
supimlogSigma_g = np.zeros(im_canvas.shape)
sdoimlogSigma_g = np.zeros(im_canvas.shape)

nx, ny = im_canvas.shape
for ix in range(nx):
    print("ix", ix, " < ", nx)
    for iy in range(ny):
        Inus = []
        specindexes = []
        for ifreq in range(nfreqs):
            aim = mfreq_imhdus[ifreq][0].data
            aInu = aim[ix, iy]
            Inus.append(aInu)
        for ispecindex in range(nspecindexs):
            aspecindexmap = mfreq_specindexhdus[ispecindex][0].data
            aspecindex = aspecindexmap[ix, iy]
            specindexes.append(aspecindex)

        ZData.Inus = np.array(Inus)
        ZData.sInus = np.array(Inus) * fluxcal_accuracy
        ZData.alphas = np.array(specindexes)
        ZData.salphas = (1 / np.log(obsnu2s / obsnu1s)) * intraband_accuracy

        [names, mcmc_results,
         bestparams] = OptimM.MCMC(ZSetup, ZData, ASED, ZMerit)

        for iparam, aname in enumerate(names):
            if 'Tdust' in aname:
                imlogTdust[ix, iy] = bestparams[iparam]
                supimlogTdust[ix, iy] = mcmc_results[iparam][1]
                sdoimlogTdust[ix, iy] = mcmc_results[iparam][2]
            if 'dustexpo' in aname:
                imq_dustexpo[ix, iy] = bestparams[iparam]
                supimq_dustexpo[ix, iy] = mcmc_results[iparam][1]
                sdoimq_dustexpo[ix, iy] = mcmc_results[iparam][2]
            if 'amax' in aname:
                imlogamax[ix, iy] = bestparams[iparam]
                supimlogamax[ix, iy] = mcmc_results[iparam][1]
                sdoimlogamax[ix, iy] = mcmc_results[iparam][2]
            if 'Sigma_g' in aname:
                imlogSigma_g[ix, iy] = bestparams[iparam]
                supimlogSigma_g[ix, iy] = mcmc_results[iparam][1]
                sdoimlogSigma_g[ix, iy] = mcmc_results[iparam][2]

punchmap(imlogTdust, hdu_canvas, fileout=outputdir + 'imlogTdust.fits')
punchmap(supimlogTdust, hdu_canvas, fileout=outputdir + 'supimlogTdust.fits')
punchmap(sdoimlogTdust, hdu_canvas, fileout=outputdir + 'sdoimlogTdust.fits')

punchmap(imq_dustexpo, hdu_canvas, fileout=outputdir + 'imq_dustexpo.fits')
punchmap(supimq_dustexpo,
         hdu_canvas,
         fileout=outputdir + 'supimq_dustexpo.fits')
punchmap(sdoimq_dustexpo,
         hdu_canvas,
         fileout=outputdir + 'sdoimq_dustexpo.fits')

punchmap(imlogamax, hdu_canvas, fileout=outputdir + 'imlogamax.fits')
punchmap(supimlogamax, hdu_canvas, fileout=outputdir + 'supimlogamax.fits')
punchmap(sdoimlogamax, hdu_canvas, fileout=outputdir + 'sdoimlogamax.fits')

punchmap(imlogSigma_g, hdu_canvas, fileout=outputdir + 'imlogSigma_g.fits')
punchmap(supimlogSigma_g,
         hdu_canvas,
         fileout=outputdir + 'supimlogSigma_g.fits')
punchmap(sdoimlogSigma_g,
         hdu_canvas,
         fileout=outputdir + 'sdoimlogSigma_g.fits')
