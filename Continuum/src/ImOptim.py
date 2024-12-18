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

#from astropy import constants as const
from astropy.constants import astropyconst20 as const
HOME = os.environ.get('HOME')
include_path = HOME + '/common/python/include'
sys.path.append(include_path)

import PyVtools.Vtools as Vtools
from ImUtils.Resamp import gridding
from ImUtils.Cube2Im import slice0
#from Gausssmooth import Gauss_filter

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/'
sys.path.append(include_path)

import Slab.Continuum.src.AModelSED as AModelSED
import Slab.Continuum.src.SEDOptim as SEDOptim

def loadapixelvalue(outputdir,SingleLOS,tag='amax'):
    ix = SingleLOS[0]
    iy = SingleLOS[1]
    filecheck=outputdir+'imlog'+tag+'.fits'
    if  os.path.exists(filecheck):
            im=fits.open(filecheck)[0].data
            refvalue = im[iy,ix]
            print("ref value>"+filecheck+" "+str(refvalue)) 
    filecheck=outputdir+'serrimlog'+tag+'.fits'
    if  os.path.exists(filecheck):
            im=fits.open(filecheck)[0].data
            refvalue = im[iy,ix]
            print("ref value>"+filecheck+" "+str(refvalue))
    return


def load_imagfile(file_data, zoomfactor=1., Debug=False, outputdir=''):

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

    i0 = (hdr['CRPIX1'] - 1) / zoomfactor
    j0 = (hdr['CRPIX2'] - 1) / zoomfactor
    hdr['CRPIX1'] = i0 + 1
    hdr['CRPIX2'] = j0 + 1

    nx = np.rint(((hdr0['NAXIS1'] - 1) / zoomfactor) + 1)
    ny = np.rint(((hdr0['NAXIS2'] - 1) / zoomfactor) + 1)
    hdr['NAXIS1'] = nx
    hdr['NAXIS2'] = ny

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

    StoreResamp = True
    if StoreResamp:
        basefilename = os.path.basename(file_data)
        fileresamp = re.sub('.fits', '_resamp.fits', basefilename)
        print("StoreResamp punching ", outputdir + fileresamp)
        hdu.writeto(outputdir + fileresamp, overwrite=True)

    omega_beam = (np.pi / (4 * np.log(2))) * (hdr['BMAJ'] *
                                              hdr['BMIN']) * (np.pi / 180)**2
    return rrs, hdu, pixscale, omega_beam


def punchmap(im, hdu, units='', fileout='test.fits'):
    hdr = hdu[0].header
    hdr['BUNIT'] = units
    hdu[0].header = hdr
    hdu[0].data = im
    print("punching ", fileout)
    hdu.writeto(fileout, overwrite=True)


def exec_optim_1los(pos, OptimM=None, ZSetup=None, ZSED=None, ZMerit=None):
    AData = pos[2]

    #OptimM.domain = OptimM.domain_CG
    #[names, bestparams] = OptimM.ConjGrad(ZSetup, AData, ASED, ZMerit)
    #passout = [names, bestparams]

    #ASED = AModelSED.MSED(ZSetup)
    #ASED.copy(ZSED) DEV
    ASED = deepcopy(ZSED)
    if 'emcee' in OptimM.sampler:
        #OptimM.domain = OptimM.domain_MCMC
        [names, mcmc_results, bestparams, modelInus, modelalphas,
         achi2] = OptimM.MCMC(ZSetup, AData, ASED, ZMerit)
    elif 'dynesty' in OptimM.sampler:
        [names, mcmc_results, bestparams, modelInus, modelalphas,
         achi2] = OptimM.dynesty(ZSetup, AData, ASED, ZMerit)
        
    if OptimM.RunConjGrad:
        OptimM.Inherit_Init = True
        ZSetup4Powell = deepcopy(ZSetup)
        ZSetup4Powell.GoInterp = False
        [names, result_ml, modelInus, modelalphas,
         Powellchi2] = OptimM.ConjGrad(ZSetup, AData, ASED, ZMerit)
        if (Powellchi2 < achi2):
            if ZSED.Verbose:
                print("Powell improved!")

            bestparams = result_ml
            achi2 = Powellchi2

    passout = [
        pos, names, mcmc_results, bestparams, modelInus, modelalphas, achi2
    ]
    return passout


def loaddata(files_images,
             files_specindex=None,
             file_fillfactor=None,
             files_errspecindex=None,
             zoomfactor=8,
             outputdir=''):

    mfreq_imhdus = []
    omega_beams = []
    pixscaleref = None
    for afile in files_images:
        rrs, hdu, pixscale, omega_beam = load_imagfile(afile,
                                                       zoomfactor=zoomfactor,
                                                       outputdir=outputdir)
        mfreq_imhdus.append(hdu)
        omega_beams.append(omega_beam)
        if pixscaleref is not None:
            if (np.fabs(pixscale - pixscaleref) > 1E-3):
                sys.exit("align images first")
            else:
                pixscaleref = pixscale
    domega_beams = omega_beams - np.roll(omega_beams,1)
    if (np.fabs(np.max(domega_beams)) > 1E-4 * omega_beams[0]):
        sys.exit("smooth images to common beam")
                
    print(len(mfreq_imhdus))
    hdu_canvas = mfreq_imhdus[0]

    hdu_canvas.writeto(outputdir + 'canvas.fits', overwrite=True)
    # Vtools.View(hdu_canvas)

    if files_specindex is not None:
        mfreq_specindexhdus = []
        for afile in files_specindex:
            rrs2, hdu, pixscale, omega_beam_b = load_imagfile(
                afile, zoomfactor=zoomfactor, outputdir=outputdir)
            mfreq_specindexhdus.append(hdu)

        mfreq_errspecindexhdus = []
        for afile in files_errspecindex:
            rrs2, hdu, pixscale, omega_beam_b = load_imagfile(
                afile, zoomfactor=zoomfactor, outputdir=outputdir)
            mfreq_errspecindexhdus.append(hdu)

        print(len(mfreq_specindexhdus))
        retvals = [
            hdu_canvas, mfreq_imhdus, mfreq_specindexhdus,
            mfreq_errspecindexhdus, omega_beams
        ]
    else:
        retvals = [hdu_canvas, mfreq_imhdus, omega_beams]

    if file_fillfactor is not None:
        rrs, hdu, pixscale, omega_beam = load_imagfile(file_fillfactor,
                                                       zoomfactor=zoomfactor,
                                                       outputdir=outputdir)
        im_f = hdu[0].data

        retvals.append(im_f)

    return retvals


def exec_imoptim(
        OptimM,
        ZSetup,
        ZData,
        ZSED,
        ZMerit,
        hdu_canvas,
        mfreq_imhdus,
        im_fillfactor=None,
        mfreq_specindexhdus=None,
        mfreq_errspecindexhdus=None,
        n_cores_map=4,
        intensity_threshold=[0, 5],  # ifreq, nthres
        files_images=None,
        files_specindex=None,  # only required for naming
        omega_beams=[],
        fluxcal_accuracy=[],
        shift_fluxcal=None,
        SingleLOS=None,
        intraband_accuracy=0.008):
    """
    Two options to set image masks:
    1- with a filling factor image passed as im_fillfactor
    2- with an intensity mask, setup with the 2 elt array intensity_threshold. [0] is the frequency index, [1] is the threshold in units of the noise
    """

    if SingleLOS is None:
        ZSetup.Verbose = False
        ZSED.Verbose = False

    nfreqs = len(mfreq_imhdus)
    im_canvas = hdu_canvas[0].data
    outputdir = ZSetup.outputdir
    rmsnoises = ZData.rmsnoises

    #rmsnoises_nu1s = ZData.rmsnoises_nu1s
    #rmsnoises_nu2s = ZData.rmsnoises_nu2s

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
    chi2map = np.zeros(im_canvas.shape)

    intensitymask = np.zeros(im_canvas.shape, dtype=int)

    modelimages = []
    for ifreq in range(nfreqs):
        amodelimage = np.zeros(im_canvas.shape)
        modelimages.append(amodelimage)

    if ZMerit.with_specindexdata:
        nspecindexs = len(mfreq_specindexhdus)
        obsnu2s = ZData.nu2s_alphas
        obsnu1s = ZData.nu1s_alphas
        modelspecindexs = []
        for ispecindex in range(nspecindexs):
            amodelspecindex = np.zeros(im_canvas.shape)
            modelspecindexs.append(amodelspecindex)

    tasks = []
    ny, nx = im_canvas.shape
    hdr_canvas = hdu_canvas[0].header
    ivec = np.arange(0, nx)
    jvec = np.arange(0, ny)
    iis, jjs = np.meshgrid(ivec, jvec)
    xxs = hdr_canvas['CDELT1'] * 3600. * (iis - (hdr_canvas['CRPIX1'] - 1))
    yys = hdr_canvas['CDELT2'] * 3600. * (jjs - (hdr_canvas['CRPIX2'] - 1))
    rrs = np.sqrt(xxs**2 + yys**2)

    fluxcal_factors = np.ones(nfreqs)
    if shift_fluxcal is not None:
        for ifreq in range(nfreqs):
            asigmadev = shift_fluxcal[ifreq]
            afactor = np.random.normal(1., asigmadev, 1)
            fluxcal_factors[ifreq] = afactor
        print("SHIFTING FLUX CALS BY: ", fluxcal_factors)
        np.savetxt(outputdir + 'fluxcal_factors.txt', fluxcal_factors)

        if ZMerit.with_specindexdata:
            sigma_intraband_specindex = np.log(1. +
                                               intraband_accuracy) / np.log(
                                                   obsnu2s / obsnu1s)
            shift_intraband_specindex = np.random.normal(
                0., sigma_intraband_specindex, len(obsnu1s))
            print("SHIFTING intrabandspecindex by : ",
                  shift_intraband_specindex)

            
    for ix in range(nx):
        for iy in range(ny):

            if SingleLOS is not None:
                if not ((ix == SingleLOS[0]) & (iy == SingleLOS[1])):
                    continue
                #print("ix ", ix, " iy ", iy)
                dalpha = xxs[iy, ix]
                ddelta = yys[iy, ix]
                OptimM.label4SED = r"$\Delta\alpha=%.2f~~\Delta\delta=%.2f$" % (
                    dalpha, ddelta)
                print("OptimM.label4SED",OptimM.label4SED)
                #OptimM.label4SED = r"$\Delta\alpha=%.2f$" % (
                #    dalpha) + "\n" + r"$\Delta\delta=%.2f$" % (ddelta)
                #print("alpha :", xxs[iy, ix], "delta:", yys[iy, ix], "radius",
                #      rrs[iy, ix])
                hdu_dum = deepcopy(hdu_canvas)
                hdu_dum[0].data = xxs
                hdu_dum.writeto(ZSetup.outputdir + 'xxs.fits', overwrite=True)
                hdu_dum[0].data = yys
                hdu_dum.writeto(ZSetup.outputdir + 'yys.fits', overwrite=True)
                hdu_dum[0].data = im_canvas
                hdu_dum.writeto(ZSetup.outputdir + 'im_canvas_dum.fits',
                                overwrite=True)
                hdu_dum.close()
                #print("im_canvas.shape",im_canvas.shape)
                #print("xxs.shape",xxs.shape)
                #print("yys.shape",yys.shape)
                #print("test values", im_canvas[40,37])
                #sys.exit()

            Inus = []
            specindexes = []
            recordfreqs = []
            fillfactor = None
            for ifreq in range(nfreqs):
                aim = mfreq_imhdus[ifreq][0].data
                ahdr = mfreq_imhdus[ifreq][0].header
                aFREQ = int(ahdr['RESTFRQ'] / 1E9)
                #print("aim[iy, ix]", aim[iy, ix])
                #Vtools.View(aim)
                aInu = aim[iy, ix] / omega_beams[ifreq]
                Inus.append(aInu)
                recordfreqs.append(aFREQ)

            Inus = np.array(Inus)

            recordfreqs = np.array(recordfreqs)
            if im_fillfactor is not None:
                #fill_factor = im_fillfactor[iy, ix]
                fill_factor = im_fillfactor[iy, ix]
                #print("fill_factor", fill_factor)
                #Vtools.View(im_fillfactor)
                if ((fill_factor == 0.) or (fill_factor >= 1.)):
                    if SingleLOS is not None:
                        print("fill_factor is ", fill_factor)
                    continue
                #print("fill_factor", fill_factor)
                Inus /= fill_factor
                intensitymask[iy, ix] = 1
            else:
                ifreq_thresh = intensity_threshold[0]
                nthresh = intensity_threshold[1]
                if (Inus[ifreq_thresh] < nthresh * rmsnoises[ifreq_thresh]):
                    if SingleLOS is not None:
                        print("below noise threshold, intensity is",
                              Inus[ifreq_thresh], " noise is: ", rmsnoises[0])
                    continue
                intensitymask[iy, ix] = 1

            if shift_fluxcal is not None:
                Inus *= fluxcal_factors

            if ZMerit.with_specindexdata:
                errspecindexes = []
                Inu1s = []
                nu1s = []
                nu2s = []
                for ispecindex in range(nspecindexs):
                    aspecindexmap = mfreq_specindexhdus[ispecindex][0].data
                    aspecindexhdr = mfreq_specindexhdus[ispecindex][0].header
                    anu1 = int(aspecindexhdr['FREQ1'] / 1E9)
                    anu2 = int(aspecindexhdr['FREQ2'] / 1E9)
                    if (anu2 < anu1):
                        sys.exit("wrong order for specindexes")
                    inu1 = np.argmin(np.fabs(recordfreqs - anu1))
                    inu2 = np.argmin(np.fabs(recordfreqs - anu2))
                    Inu1s.append(Inus[inu1])
                    nu1s.append(anu1)
                    nu2s.append(anu2)
                    aspecindex = aspecindexmap[iy, ix]
                    specindexes.append(aspecindex)
                    aerrspecindexmap = mfreq_errspecindexhdus[ispecindex][
                        0].data
                    aerrspecindex = aerrspecindexmap[iy, ix]
                    errspecindexes.append(aerrspecindex)

                nu1s = np.array(nu1s)
                nu2s = np.array(nu2s)

            #AData = SEDOptim.Data()
            #AData.copy(ZData)
            AData=deepcopy(ZData)
            AData.Inus = Inus
            AData.fillfactor = fillfactor
            if rmsnoises is not None:
                #if shift_fluxcal is not None:
                AData.sInus = np.sqrt(rmsnoises**2 +
                                      (np.array(Inus) * fluxcal_accuracy)**2)
                #else:
                #    AData.sInus = rmsnoises
            else:
                AData.sInus = np.array(Inus) * fluxcal_accuracy

            if ZMerit.with_specindexdata:
                AData.alphas = np.array(
                    specindexes) + shift_intraband_specindex
                AData.Inu1s = np.array(Inu1s)  # AData.Inus.copy()
                if shift_fluxcal is not None:
                    AData.salphas = np.array(errspecindexes)
                else:
                    AData.salphas = np.array(errspecindexes) + (
                        np.log(1. + intraband_accuracy) / np.log(nu2s / nu1s))

            tasks.append([ix, iy, AData])

    #        {'ZSetup': ZSetup,'ASED': ASED,'ZMerit': ZMerit}

    print("loaded all ", len(tasks), "tasks")

    if n_cores_map > 1: 
        with Pool(n_cores_map) as pool:
            Pooloutput = list(
                tqdm(pool.imap(
                    partial(exec_optim_1los,
                            OptimM=OptimM,
                            ZSetup=ZSetup,
                            ZSED=ZSED,
                            ZMerit=ZMerit), tasks),
                     total=len(tasks)))
            pool.close()
            pool.join()
    else:
        Pooloutput=[]
        for atask in tasks:
            print("atask",atask)
            Pooloutput.append(exec_optim_1los(atask,OptimM=OptimM, ZSetup=ZSetup, ZSED=ZSED, ZMerit=ZMerit))

            
    if SingleLOS is not None:
        loadapixelvalue(outputdir,SingleLOS,tag='amax')
        loadapixelvalue(outputdir,SingleLOS,tag='Sigma_g')
        loadapixelvalue(outputdir,SingleLOS,tag='Tdust')
        return

    for alos in Pooloutput:
        ix = alos[0][0]
        iy = alos[0][1]
        names = alos[1]
        mcmc_results = alos[2]
        bestparams = alos[3]
        modelInus = alos[4]
        modelalphas = alos[5]
        achi2 = alos[6]

        chi2map[iy, ix] = achi2

        for ifreq in range(nfreqs):
            modelimages[ifreq][iy, ix] = modelInus[ifreq] * ZData.omega_beam

        if ZMerit.with_specindexdata:
            for ispecindex in range(nspecindexs):
                modelspecindexs[ispecindex][iy, ix] = modelalphas[ispecindex]

        for iparam, aname in enumerate(names):
            if 'Tdust' in aname:
                if OptimM.MCMCresult_UseMedian:
                    imlogTdust[iy, ix] = mcmc_results[iparam][0]
                else:
                    imlogTdust[iy, ix] = bestparams[iparam]
                supimlogTdust[iy, ix] = mcmc_results[iparam][1]
                sdoimlogTdust[iy, ix] = mcmc_results[iparam][2]
            if 'dustexpo' in aname:
                if OptimM.MCMCresult_UseMedian:
                    imq_dustexpo[iy, ix] = mcmc_results[iparam][0]
                else:
                    imq_dustexpo[iy, ix] = bestparams[iparam]
                supimq_dustexpo[iy, ix] = mcmc_results[iparam][1]
                sdoimq_dustexpo[iy, ix] = mcmc_results[iparam][2]
            if 'amax' in aname:
                if OptimM.MCMCresult_UseMedian:
                    imlogamax[iy, ix] = mcmc_results[iparam][0]
                else:
                    imlogamax[iy, ix] = bestparams[iparam]
                supimlogamax[iy, ix] = mcmc_results[iparam][1]
                sdoimlogamax[iy, ix] = mcmc_results[iparam][2]
            if 'Sigma_g' in aname:
                if OptimM.MCMCresult_UseMedian:
                    imlogSigma_g[iy, ix] = mcmc_results[iparam][0]
                else:
                    imlogSigma_g[iy, ix] = bestparams[iparam]
                supimlogSigma_g[iy, ix] = mcmc_results[iparam][1]
                sdoimlogSigma_g[iy, ix] = mcmc_results[iparam][2]

    for ifreq in range(nfreqs):
        datafile = os.path.basename(files_images[ifreq])
        modelfile = re.sub('.fits', '_model.fits', datafile)
        punchmap(modelimages[ifreq], hdu_canvas, fileout=outputdir + modelfile)

    if ZMerit.with_specindexdata:
        for ispecindex in range(nspecindexs):
            datafile = os.path.basename(files_specindex[ispecindex])
            modelfile = re.sub('.fits', '_model.fits', datafile)
            punchmap(modelspecindexs[ispecindex],
                     hdu_canvas,
                     fileout=outputdir + modelfile)

    punchmap(intensitymask,
             hdu_canvas,
             fileout=outputdir + 'intensitymask.fits')

    punchmap(imlogTdust, hdu_canvas, fileout=outputdir + 'imlogTdust.fits')
    punchmap(supimlogTdust,
             hdu_canvas,
             fileout=outputdir + 'supimlogTdust.fits')
    punchmap(sdoimlogTdust,
             hdu_canvas,
             fileout=outputdir + 'sdoimlogTdust.fits')
    punchmap((sdoimlogTdust + supimlogTdust) / 2.,
             hdu_canvas,
             fileout=outputdir + 'serrimlogTdust.fits')

    punchmap(imq_dustexpo, hdu_canvas, fileout=outputdir + 'imq_dustexpo.fits')
    punchmap(supimq_dustexpo,
             hdu_canvas,
             fileout=outputdir + 'supimq_dustexpo.fits')
    punchmap(sdoimq_dustexpo,
             hdu_canvas,
             fileout=outputdir + 'sdoimq_dustexpo.fits')
    punchmap((sdoimq_dustexpo + supimq_dustexpo) / 2.,
             hdu_canvas,
             fileout=outputdir + 'serrimq_dustexpo.fits')

    punchmap(imlogamax, hdu_canvas, fileout=outputdir + 'imlogamax.fits')
    punchmap(supimlogamax, hdu_canvas, fileout=outputdir + 'supimlogamax.fits')
    punchmap(sdoimlogamax, hdu_canvas, fileout=outputdir + 'sdoimlogamax.fits')
    punchmap((sdoimlogamax + supimlogamax) / 2.,
             hdu_canvas,
             fileout=outputdir + 'serrimlogamax.fits')

    punchmap(imlogSigma_g, hdu_canvas, fileout=outputdir + 'imlogSigma_g.fits')
    punchmap(supimlogSigma_g,
             hdu_canvas,
             fileout=outputdir + 'supimlogSigma_g.fits')
    punchmap(sdoimlogSigma_g,
             hdu_canvas,
             fileout=outputdir + 'sdoimlogSigma_g.fits')
    punchmap((sdoimlogSigma_g + supimlogSigma_g) / 2.,
             hdu_canvas,
             fileout=outputdir + 'serrimlogSigma_g.fits')

    punchmap(chi2map, hdu_canvas, fileout=outputdir + 'chi2map.fits')
