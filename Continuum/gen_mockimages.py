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
import scipy as sp
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

import PolarFuncs


def load_canvas(file_canvas, zoomfactor=1., Debug=False):

    #f0 = fits.open(file_canvas)
    hdu_canvas = slice0(file_canvas, ReturnHDUList=True)
    im_canvas0 = hdu_canvas[0].data
    hdr_canvas0 = hdu_canvas[0].header

    pixscale = hdr_canvas0['CDELT2'] * 3600.
    print("canvas pixscale = ", pixscale)

    hdr_canvas = deepcopy(hdr_canvas0)
    hdr_canvas['CDELT1'] *= zoomfactor
    hdr_canvas['CDELT2'] *= zoomfactor

    f1 = gridding(hdu_canvas, hdr_canvas, ReturnHDUList=True)
    im_canvas = f1[0].data

    pixscale = hdr_canvas['CDELT2'] * 3600.

    print("pixscale = ", pixscale)

    (ny, nx) = im_canvas.shape
    ivec = np.arange(0, nx)
    jvec = np.arange(0, ny)
    ii, jj = np.meshgrid(ivec, jvec)

    i0 = int((nx - 1) / 2)
    j0 = int((ny - 1) / 2)

    hdr_canvas['CRPIX1'] = i0 + 1
    hdr_canvas['CRPIX2'] = j0 + 1

    dxxs = -pixscale * (ii - i0)
    dyys = pixscale * (jj - i0)
    rrs = np.sqrt((dxxs)**2 + (dyys)**2)

    hdr_canvas['PIXSCALE'] = pixscale
    hdu_canvas[0].data = im_canvas
    hdu_canvas[0].hdr_canvas = hdr_canvas

    return rrs, hdu_canvas, pixscale


def ftaper_gap(r, r1, pedestal, expo=5.):
    taper = (pedestal + ((np.tanh((r - r1) / r1))**expo) * (1. - pedestal))
    return taper


def ftaper(r, r1):
    taper = (1. - np.tanh((r - r1) / r1))**4.
    return taper


def z_func_gap(r, z0, r0, q, r1, r2, pedestal):
    taper_trunc = 1.
    powerdisk = 1.
    if (isinstance(r, np.ndarray)):
        taper_trunc = np.ones(r.shape)
        alltaper = ftaper(r, r2)
        taper_trunc[(r > r2)] = alltaper[(r > r2)]
    else:
        if (r > r2):
            taper_trunc = ftaper(r, r2)
    if (isinstance(r, np.ndarray)):
        powerdisk = np.ones(r.shape)
        powerdisk[(r < r0)] = (r[(r < r0)] / r0)**q
    else:
        if (r < r0):
            powerdisk = (r / r0)**q

    if (isinstance(r, np.ndarray)):
        taper_gap = np.ones(r.shape)
        alltaper = ftaper_gap(r, r1, pedestal, expo=4)
        taper_gap[(r < r1)] = pedestal
        taper_gap[(r >= r1)] = alltaper[(r >= r1)]
    else:
        if (r < r1):
            taper_gap = pedestal
        else:
            taper_gap = ftaper_gap(r, r1, pedestal, expo=4)

    return z0 * powerdisk * taper_gap * taper_trunc


def z_func(r, z0, r0, r1, q):
    taper = 1.
    if (isinstance(r, np.ndarray)):
        taper = np.ones(r.shape)
        alltaper = ftaper(r, r1)
        taper[(r > r1)] = alltaper[(r > r1)]
    else:
        if (r > r1):
            taper = ftaper(r, r1)
    return (z0 * (r / r0)**q) * taper


def get_im(
        profile,
        hdu_canvas,
        hdupolar,
        units='K',
        Smooth=False,
        AddNoise=False,  # umJy/beam
        restfreq=False,
        outputdir='./mockdata/',
        fileout='Tdust.fits'):

    im_canvas_polar = hdupolar.data
    im_polar = np.zeros(im_canvas_polar.shape)
    im_polar[:, :] = profile[:, np.newaxis]
    im = PolarFuncs.polartocart(im_polar)
    hdu = deepcopy(hdu_canvas)
    hdr = hdu[0].header
    hdr['BUNIT'] = units
    if restfreq:
        hdr['RESTFRQ'] = restfreq
    Omegabeam_pix = 1.
    if Smooth:
        sigma_x = hdr_canvas['BMAJ'] / (2. * np.sqrt(2. * np.log(2)) *
                                        hdr_canvas['CDELT2'])
        sigma_y = hdr_canvas['BMIN'] / (2. * np.sqrt(2. * np.log(2)) *
                                        hdr_canvas['CDELT2'])
        Omegabeam_pix = (np.pi / (4. * np.log(2))) * (
            hdr_canvas['BMAJ'] / hdr_canvas['CDELT2']) * (hdr_canvas['BMIN'] /
                                                          hdr_canvas['CDELT2'])
    if AddNoise:
        noisedisp = AddNoise * 1E-6 * np.sqrt(Omegabeam_pix)
        im_witness = np.zeros(im.shape)
        im_witness += np.random.normal(0., noisedisp, size=im.shape)
        #hdu[0].data = im_witness
        #hdu.writeto('noise_witness.fits', overwrite=True)
        if Smooth:
            im_witnesss = Gauss_filter(im_witness, sigma_x, sigma_y, 0.)
            netnoise = np.std(im_witnesss)
            print("netnoise", netnoise, "target noise", AddNoise)
            noisedisp2 = noisedisp * (AddNoise * 1E-6 / netnoise)
        else:
            noisedisp2 = noisedisp

        im_witness = np.ones(im.shape)
        im_witness += np.random.normal(0., noisedisp2, size=im.shape)
        im = im + np.random.normal(0., noisedisp2, size=im.shape)
    if Smooth:
        ims = Gauss_filter(im, sigma_x, sigma_y, 0.)
        im = ims
        if AddNoise:
            im_spike = np.zeros(im.shape)
            im_spike[int(hdr['CRPIX1']), int(hdr['CRPIX2'])] = 1.
            im_witnesss = Gauss_filter(im_witness, sigma_x, sigma_y, 0.)
            im_spikes = Gauss_filter(im_spike, sigma_x, sigma_y, 0.)
            hdu[0].data = im_witnesss
            hdu.writeto(outputdir + 'snoise_witness.fits', overwrite=True)
            hdu[0].data = im_spikes
            hdu.writeto(outputdir + 'spike_witness.fits', overwrite=True)

    hdu[0].data = im
    hdu[0].header = hdr
    hdu.writeto(outputdir + fileout, overwrite=True)
    return hdu


def extract_profile(hdu_im, rs, outputdir='./mockdata/', fileout='I1.dat'):

    hdr = hdu[0].header
    if 'BMAJ' in hdr.keys():
        bmaj = hdr['BMAJ']
        Nind = 2. * np.pi * rs / (bmaj * 3600.)
        Nind[(Nind < 1.)] = 1.
    else:
        Nind = np.ones(rs.shape)

    im = hdu_im[0].data
    hdr = hdu_im[0].header

    im_polar = PolarFuncs.carttopolar(im)
    #Vtools.View(im_polar)
    #Iprof = np.median(im_polar, axis=1)
    Iprof = np.average(im_polar, axis=1)
    dispIprof = np.std(im_polar, axis=1)
    sIprof = dispIprof / np.sqrt(Nind)

    if 'RESTFRQ' in hdr.keys():
        fileout = 'Iprof_' + str(hdr['RESTFRQ'] / 1E9) + '.dat'

    save_prof = np.zeros((len(rs), 4))
    save_prof[:, 0] = rs
    save_prof[:, 1] = Iprof
    save_prof[:, 2] = sIprof
    save_prof[:, 3] = dispIprof

    #Vtools.Spec([save_prof[:, 0:2]])

    np.savetxt(outputdir + fileout, save_prof)


######################################################################
######################################################################
######################################################################

outputdir = './mockdata/'
#outputdir = './mockdata_nothermalnoise/'
os.system("mkdir " + outputdir)

rrs, hdu_canvas, pixscale = load_canvas(
    './data/tclean_HD135344Bbriggs2.0_self.fits')

im_canvas = hdu_canvas[0].data
hdr_canvas = hdu_canvas[0].header
resolution = 0.040  # arcsec
hdr_canvas['BMAJ'] = resolution / (3600.)
hdr_canvas['BMIN'] = resolution / (3600.)
omega_beam = (np.pi / (4. * np.log(2))) * (resolution * np.pi /
                                           (180. * 3600.))**2  # C10 B3 beam

im_canvas_polar = PolarFuncs.carttopolar(im_canvas)
nphis, nrs = im_canvas_polar.shape
Debug = True
if Debug:
    print("nphis", nphis)
    print("nrs", nrs)
hdupolar = fits.PrimaryHDU()
hdrpolar = hdupolar.header
hdrpolar['NAXIS1'] = nrs
hdrpolar['NAXIS2'] = nphis
hdrpolar['CRPIX1'] = 1
hdrpolar['CRVAL1'] = 0.
hdrpolar['CDELT1'] = 2. * np.pi / (float(nphis) - 1.)  # DEV DEV DEV
hdrpolar['CRPIX2'] = 1
hdrpolar['CRVAL2'] = 0.
hdrpolar['CDELT2'] = hdr_canvas['CDELT2']
hdupolar.header = hdrpolar
hdupolar.data = im_canvas_polar

pixscale = 0.04  # arcsec
pixscale = hdrpolar['CDELT2'] * 3600.  # arcsec

origin_offset = pixscale
rs = pixscale * (np.arange(hdrpolar['NAXIS2']) -
                 (hdrpolar['CRPIX2'] - 1)) + origin_offset

######################################################################
pars = {'z0': 2., 'r0': 0.4, 'q': 6, 'r1': 0.45, 'r2': 0.8, 'pedestal': 0.02}
Sigma_g = z_func_gap(rs, pars['z0'], pars['r0'], pars['q'], pars['r1'],
                     pars['r2'], pars['pedestal'])

Sigma_g = 100. * Sigma_g / np.max(Sigma_g)

modprof1 = np.zeros((len(rs), 2))
modprof1[:, 0] = rs - origin_offset
modprof1[:, 1] = Sigma_g
np.savetxt(outputdir + 'Sigma_g_profile.dat', modprof1)
print("model Sigma_g(R)")
#Vtools.Spec([modprof1])

get_im(Sigma_g,
       hdu_canvas,
       hdupolar,
       units='K',
       outputdir=outputdir,
       fileout='Sigma_g.fits')

######################################################################
pars = {'z0': 2., 'r0': 0.4, 'q': 6, 'r1': 0.45, 'r2': 0.8, 'pedestal': 0.0}
amax = z_func_gap(rs, pars['z0'], pars['r0'], pars['q'], pars['r1'],
                  pars['r2'], pars['pedestal'])

dumamax = amax / max(amax)
amax0 = 0.0001  # 1micron
amax = amax0 + dumamax * (max(dumamax) - amax0)
print("global lmaximum grain: ", max(amax))

modprof2 = np.zeros((len(rs), 2))
modprof2[:, 0] = rs - origin_offset
modprof2[:, 1] = amax
print("model amax")
#Vtools.Spec([modprof1, modprof2])
#Vtools.Spec([modprof2])
np.savetxt(outputdir + 'amax_profile.dat', modprof2)

get_im(amax,
       hdu_canvas,
       hdupolar,
       units='cm',
       outputdir=outputdir,
       fileout='amax.fits')

######################################################################
qdustexpo = amax**2 * max(amax) / max(amax**2)
qdustexpo /= max(qdustexpo)
qdustexpo0 = -3.5
qdustexpo1 = -2.5
qdustexpo = qdustexpo0 + qdustexpo * (qdustexpo1 - qdustexpo0)
modprof3 = np.zeros((len(rs), 2))
modprof3[:, 0] = rs - origin_offset
modprof3[:, 1] = qdustexpo
print("model qdustexpo")
#Vtools.Spec([
#    modprof1,
#    modprof2,
#    modprof3,
#])
#Vtools.Spec([
#    modprof3,
#])
np.savetxt(outputdir + 'qdustexpo_profile.dat', modprof3)

get_im(qdustexpo,
       hdu_canvas,
       hdupolar,
       units='',
       outputdir=outputdir,
       fileout='qdustexpo.fits')

######################################################################
Tdust0 = 30.
r_Tdust0 = 1.

Tdust = Tdust0 * (rs / r_Tdust0)**-0.5
modprof = np.zeros((len(rs), 2))
modprof[:, 0] = rs - origin_offset
modprof[:, 1] = Tdust
print("model Tdust")
#Vtools.Spec([
#    modprof,
#])
np.savetxt(outputdir + 'Tdust_profile.dat', modprof)

get_im(Tdust,
       hdu_canvas,
       hdupolar,
       units='K',
       outputdir=outputdir,
       fileout='Tdust.fits')

######################################################################
######################################################################
######################################################################
# mock multifreq data

#obsfreqs_alphas = np.array(
#    [100E9, 130E9, 150E9, 180E9, 230E9, 260E9, 345E9, 375E9])

# https://almascience.eso.org/proposing/sensitivity-calculator
# 2nd octile conditions, 1h integration

obsfreqs = np.array([100E9, 150E9, 230E9, 345E9, 694E9])
rmsnoise = 0.5 * np.array([9., 9.5, 12, 21.6, 313])  #rms noise in uJy/beam

obsfreqs_alphas = np.array(
    [100E9, 130E9, 150E9, 165E9, 230E9, 245E9, 345E9, 360E9])
rmsnoise_alphas = 0.5 * np.array([9., 9., 9.5, 9.5, 12, 12, 21.6, 21.6
                                  ])  #rms noise in uJy/beam

ZSetup = AModelSED.Setup(
    filetag='',  # False
    PrintChi2s=True,
    ClearOutputDir=False,
    GenFigs=False,
    opct_file='opct_mix.txt',
    VerboseInit=False,
    outputdir=outputdir)

#ZSED = AModelSED.MSED(
#    ZSetup,
#    Tdust=20.,
#    q_dustexpo=-3.5,
#    f_grain=1.,  # grain filling factor
#    amin=1E-4,  # cm
#    amax=1.,  # cm, maximum grain size
#    Sigma_g=100. * 0.5,  # g/cm2
#    gtod_ratio=100.,
#    rho0=2.77,  # g/cm3
#    N_asizes=1000,
#    nus=obsfreqs)

allfreqs = obsfreqs
allrmss = rmsnoise
for ifreq1, afreq1 in enumerate(obsfreqs_alphas):
    if not (afreq1 in allfreqs):
        allfreqs = np.append(allfreqs, afreq1)
        allrmss = np.append(allrmss, rmsnoise_alphas[ifreq1])

print("allfreqs", allfreqs)

nfreqs = len(allfreqs)
obs_profiles = np.zeros([nfreqs, len(rs)])

for ir, ar in enumerate(rs):
    ASED = AModelSED.MSED(
        ZSetup,
        Tdust=Tdust[ir],
        q_dustexpo=qdustexpo[ir],
        f_grain=1.,  # grain filling factor
        amin=1E-5,  # cm
        amax=amax[ir],  # cm, maximum grain size
        Sigma_g=Sigma_g[ir],  # g/cm2
        gtod_ratio=100.,
        rho0=2.77,  # g/cm3
        N_asizes=400,
        nus=allfreqs)
    ASED.calcul()
    obs_profiles[:, ir] = ASED.Inus[:]

for ifreq, afreq in enumerate(allfreqs):
    armsnoise = allrmss[ifreq]
    aprofile = omega_beam * obs_profiles[ifreq, :]
    aname = "I_%d.fits" % (afreq / 1E9)
    hdu = get_im(
        aprofile,
        hdu_canvas,
        hdupolar,
        units='Jy/beam',
        outputdir=outputdir,
        restfreq=afreq,
        AddNoise=armsnoise,  # umJy/beam
        Smooth=True,
        fileout=aname)
    extract_profile(hdu, rs, outputdir=outputdir)

npairs = int(len(obsfreqs_alphas) / 2)
intraband_accuracy = 0.008
for ipair in range(npairs):
    nu1 = obsfreqs_alphas[int(ipair * 2)]
    nu2 = obsfreqs_alphas[int(ipair * 2 + 1)]
    aname_1 = "I_%d.fits" % (nu1 / 1E9)
    aname_2 = "I_%d.fits" % (nu2 / 1E9)
    hdu1 = fits.open(outputdir + aname_1)
    im1 = hdu1[0].data
    hdu2 = fits.open(outputdir + aname_2)
    im2 = hdu2[0].data
    specindex = np.log(im1 / im2) / np.log(nu1 / nu2)

    hdu1[0].data = specindex
    hdr1 = hdu1[0].header
    hdr1['BUNIT'] = ''
    hdu1[0].header = hdr1
    aname_1 = "specindec_%d.fits" % (nu1 / 1E9)
    hdu1.writeto(outputdir + aname_1, overwrite=True)

    rmsnoises_nu2 = rmsnoise_alphas[int(ipair * 2)] * 1E-6
    rmsnoises_nu1 = rmsnoise_alphas[int(ipair * 2 + 1)] * 1E-6
    sigma_2 = np.sqrt((im2 * intraband_accuracy)**2 + rmsnoises_nu2**2)
    sigma_1 = rmsnoises_nu1
    sspecindex = (1 / np.log(nu2 / nu1)) * np.sqrt((sigma_2 / im2)**2 +
                                                   (sigma_1 / im1)**2)

    hdu1[0].data = sspecindex
    hdr1 = hdu1[0].header
    hdr1['BUNIT'] = ''
    hdu1[0].header = hdr1
    aname_1 = "sspecindec_%d.fits" % (nu1 / 1E9)
    hdu1.writeto(outputdir + aname_1, overwrite=True)
