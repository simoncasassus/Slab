import os
import numpy as np
import matplotlib
from astropy.io import fits
from copy import deepcopy
from functools import partial
import re

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath as cma
# from time import time,gmtime, strftime
import sys
from multiprocessing import Pool
from tqdm import tqdm

from astropy import constants as const

HOME = os.environ.get('HOME')
include_path = HOME + '/common/python/include'
sys.path.append(include_path)

import PyVtools.Vtools as Vtools
from ImUtils.Resamp import gridding
from ImUtils.Cube2Im import slice0
#from Gausssmooth import Gauss_filter

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
    hdu[0].header = hdr

    omega_beam = (np.pi / (4 * np.log(2))) * (hdr['BMAJ'] *
                                              hdr['BMIN']) * (np.pi / 180)**2
    return rrs, hdu, pixscale, omega_beam


def punchmap(im, hdu, units='', fileout='test.fits'):
    hdr = hdu[0].header
    hdr['BUNIT'] = units
    hdu[0].header = hdr
    hdu[0].data = im
    hdu.writeto(fileout, overwrite=True)


def exec_optim_1los(pos, OptimM=None, ZSetup=None, ASED=None, ZMerit=None):
    AData = pos[2]
    if OptimM.RunConjGrad:
        OptimM.domain = OptimM.domain_CG
        [names, bestparams] = OptimM.ConjGrad(ZSetup, AData, ASED, ZMerit)
        passout = [names, bestparams]

    OptimM.domain = OptimM.domain_MCMC
    [names, mcmc_results, bestparams, modelInus,
     modelalphas] = OptimM.MCMC(ZSetup, AData, ASED, ZMerit)
    passout = [pos, names, mcmc_results, bestparams, modelInus, modelalphas]
    return passout


def loaddata(files_images, files_specindex, zoomfactor=8):

    mfreq_imhdus = []
    omega_beams = []
    for afile in files_images:
        rrs, hdu, pixscale, omega_beam = load_imagfile(afile,
                                                       zoomfactor=zoomfactor)
        mfreq_imhdus.append(hdu)
        omega_beams.append(omega_beam)

    print(len(mfreq_imhdus))
    hdu_canvas = mfreq_imhdus[0]

    hdu_canvas.writeto('canvas.fits', overwrite=True)
    # Vtools.View(hdu_canvas)

    mfreq_specindexhdus = []
    for afile in files_specindex:
        rrs2, hdu, pixscale, omega_beam_b = load_imagfile(
            afile, zoomfactor=zoomfactor)
        mfreq_specindexhdus.append(hdu)

    print(len(mfreq_specindexhdus))
    return hdu_canvas, mfreq_imhdus, mfreq_specindexhdus, omega_beams


def exec_imoptim(OptimM,
                 ZSetup,
                 ZData,
                 ASED,
                 ZMerit,
                 hdu_canvas,
                 mfreq_imhdus,
                 mfreq_specindexhdus,
                 n_cores_map=4,
                 files_images=None,
                 files_specindex=None,
                 omega_beams=[],
                 fluxcal_accuracy=[],
                 SingleLOS=None,
                 intraband_accuracy=0.008):

    nfreqs = len(mfreq_imhdus)
    nspecindexs = len(mfreq_specindexhdus)
    im_canvas = hdu_canvas[0].data
    outputdir = ZSetup.outputdir
    obsnu2s = ZData.nu2s_alphas
    obsnu1s = ZData.nu1s_alphas

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

    modelimages = []
    for ifreq in range(nfreqs):
        amodelimage = np.zeros(im_canvas.shape)
        modelimages.append(amodelimage)
    modelspecindexs = []
    for ispecindex in range(nspecindexs):
        amodelspecindex = np.zeros(im_canvas.shape)
        modelspecindexs.append(amodelspecindex)

    tasks = []
    nx, ny = im_canvas.shape
    for ix in range(nx):
        for iy in range(ny):
            #if not ((ix == 16) & (iy == 16)): local gap in Tdust
            #if not ((ix == 12) & (iy == 12)): local ring in Tdust?
            if SingleLOS is not None:
                if not ((ix == SingleLOS[0]) & (iy == SingleLOS[1])):
                    continue
            #print("ix ", ix, " iy ", iy)
            Inus = []
            specindexes = []
            for ifreq in range(nfreqs):
                aim = mfreq_imhdus[ifreq][0].data
                aInu = aim[ix, iy] / omega_beams[ifreq]
                Inus.append(aInu)
            for ispecindex in range(nspecindexs):
                aspecindexmap = mfreq_specindexhdus[ispecindex][0].data
                aspecindex = aspecindexmap[ix, iy]
                specindexes.append(aspecindex)

            AData = SEDOptim.Data()
            AData.copy(ZData)
            AData.Inus = np.array(Inus)
            AData.sInus = np.array(Inus) * fluxcal_accuracy
            AData.alphas = np.array(specindexes)
            AData.salphas = (1 /
                             np.log(obsnu2s / obsnu1s)) * intraband_accuracy
            tasks.append([ix, iy, AData])
            AData.Inu1s = AData.Inus.copy()

    #        {'ZSetup': ZSetup,'ASED': ASED,'ZMerit': ZMerit}

    print("loaded all ", len(tasks), "tasks")
    
    with Pool(n_cores_map) as pool:
        Pooloutput = list(
            tqdm(pool.imap(
                partial(exec_optim_1los,
                        OptimM=OptimM,
                        ZSetup=ZSetup,
                        ASED=ASED,
                        ZMerit=ZMerit), tasks),
                 total=len(tasks)))
        pool.close()
        pool.join()

    for alos in Pooloutput:
        ix = alos[0][0]
        iy = alos[0][1]
        names = alos[1]
        mcmc_results = alos[2]
        bestparams = alos[3]
        modelInus = alos[4]
        modelalphas = alos[5]

        for ifreq in range(nfreqs):
            modelimages[ifreq][ix, iy] = modelInus[ifreq]
        for ispecindex in range(nspecindexs):
            modelspecindexs[ispecindex][ix, iy] = modelalphas[ispecindex]

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

    for ifreq in range(nfreqs):
        datafile = os.path.basename(files_images[ifreq])
        modelfile = re.sub('.fits', '_model.fits', datafile)
        punchmap(modelimages[ifreq], hdu_canvas, fileout=outputdir + modelfile)

    for ispecindex in range(nspecindexs):
        datafile = os.path.basename(files_specindex[ispecindex])
        modelfile = re.sub('.fits', '_model.fits', datafile)
        punchmap(modelspecindexs[ispecindex],
                 hdu_canvas,
                 fileout=outputdir + modelfile)

    punchmap(imlogTdust, hdu_canvas, fileout=outputdir + 'imlogTdust.fits')
    punchmap(supimlogTdust,
             hdu_canvas,
             fileout=outputdir + 'supimlogTdust.fits')
    punchmap(sdoimlogTdust,
             hdu_canvas,
             fileout=outputdir + 'sdoimlogTdust.fits')

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