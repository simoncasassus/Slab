import sys
import os
import os.path
import numpy as np
from astropy.io import fits
import time
import astropy.constants as const
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.rc('font', family='arial')
matplotlib.rcParams.update({'font.size': 12})

HOME = os.environ.get('HOME')
include_path = HOME + '/common/python/include'
sys.path.append(include_path)
include_path = HOME + '/gitcommon/Slab/Continuum/'
sys.path.append(include_path)

import PolarFuncs


def addprofile(axprofile,
               rs,
               a_min,
               a_max,
               I_prof,
               dlo=None,
               dup=None,
               linelabel='',
               axislabel='',
               label='a',
               color='grey',
               linecolor='blue',
               visibleXaxis=False,
               legendloc='best',
               y_min0=False,
               y_max0=False,
               labelloc='default',
               yscale='linear',
               ExtendRange=True,
               uperror=False,
               linestyle='solid'):

    plotmask = ((rs > a_min) & (rs < a_max))

    if dlo is not None:
        y_min = np.min(I_prof[plotmask] - dlo[plotmask])
        y_max = np.max(I_prof[plotmask] + dup[plotmask])
    else:
        y_min = np.min(I_prof[plotmask])
        y_max = np.max(I_prof[plotmask])

    if ExtendRange:
        print("extending range from ", y_min, y_max)
        if (y_min0):
            y_min = min(y_min0, y_min)
        if (y_max0):
            y_max = max(y_max0, y_max)
        print("to  ", y_min, y_max)
    else:
        y_min = y_min0
        y_max = y_max0

    axprofile.set_xlim(a_min, a_max)
    axprofile.set_ylim(y_min, y_max)

    if dlo is not None:
        axprofile.fill_between(rs[plotmask],
                               I_prof[plotmask] + dup[plotmask],
                               I_prof[plotmask] - dlo[plotmask],
                               lw=0.1,
                               color=color,
                               alpha=0.2,
                               interpolate=True)  #, step='mid'

    axprofile.plot(rs[plotmask],
                   I_prof[plotmask],
                   color=linecolor,
                   linewidth=1.5,
                   linestyle=linestyle,
                   label=linelabel)

    axprofile.legend(loc=legendloc, fontsize=12)
    axprofile.set_ylabel(axislabel)
    axprofile.set_xlabel('arcsec')
    if (yscale == 'log'):
        axprofile.text(a_min + (a_max - a_min) * 0.05, y_max * 0.8, label)
    else:
        if (labelloc == 'default'):
            axprofile.text(a_min + (a_max - a_min) * 0.05,
                           y_max - (y_max - y_min) * 0.1, label)
        else:
            axprofile.text(a_min + (a_max - a_min) * 0.05,
                           (y_max - y_min) * 0.1 + y_min, label)

    axprofile.tick_params(axis='both', length=8, direction='in', pad=10)
    axprofile.tick_params(top='on', right='on', direction='in')
    axprofile.tick_params(which='minor',
                          top='on',
                          length=4,
                          right='on',
                          direction='in')

    plt.setp(axprofile.get_xticklabels(), visible=visibleXaxis)  #, fontsize=6)

    plt.subplots_adjust(hspace=0.)

    plt.yscale(yscale)
    return y_min, y_max


def getprofile(outputdir, fileim, set_xrange=False):

    hdu = fits.open(outputdir + fileim)
    im = hdu[0].data
    hdr = hdu[0].header

    im_polar = PolarFuncs.carttopolar(im)
    nphis, nrs = im_polar.shape
    pixscale = hdr['CDELT2'] * 3600.
    rs = pixscale * (np.arange(nrs))

    if 'BMAJ' in hdr.keys():
        bmaj = hdr['BMAJ']
        Nind = 2. * np.pi * rs / (bmaj * 3600.)
        Nind[(Nind < 1.)] = 1.
    else:
        Nind = np.ones(rs.shape)

    Iprof = np.average(im_polar, axis=1)
    dispIprof = np.std(im_polar, axis=1)
    sIprof = dispIprof / np.sqrt(Nind)

    if set_xrange:
        im_polar_counts = im_polar.copy()
        im_polar_counts[(np.fabs(im_polar_counts) > 0.)] = 1.
        Nprof = np.sum(im_polar_counts, axis=1)
        valid_rs = rs[(Nprof > 0.9 * nphis)]
        r1 = np.min(valid_rs)
        r2 = np.max(valid_rs)
        return rs, Iprof, sIprof, dispIprof, r1, r2
    else:
        return rs, Iprof, sIprof, dispIprof


def proc_1param(
        axprofile,
        file_input_profile=False,  # 
        takelog=False,
        axislabel='',
        linelabel=r'$\log_{10}(T_{\rm d} \,/ \,\rm{K})$',
        outputdir='',
        ploterrors=True,
        modelimage='imlogTdust.fits',
        a_min=None,
        a_max=None,
        y_min0=None,
        y_max0=None,
        #radius_4trueparams=0.56, # Feng dust model, RING
        #radius_4trueparams=0.37, # LP dust model, RING
        radius_4trueparams=0.275, # LP dust model, GAP
        drawlegend=True,
        label='a'):

    if file_input_profile:

        
        if '.fits' in file_input_profile:
            rs, aprof, saprof, disaprof = getprofile(
                os.path.dirname(file_input_profile)+'/',
                os.path.basename(file_input_profile),
                set_xrange=False)
        else:
            (rs, aprof) = np.loadtxt(file_input_profile, unpack=True)

        if takelog:
            aprof = np.log10(aprof)

        iradius_4trueparams= np.argmin(np.fabs(rs - radius_4trueparams))
        print("true "+file_input_profile, aprof[iradius_4trueparams])

        if 'amax' in file_input_profile:
            ringamax = np.max(aprof[(rs > 0.2)])
            irad = np.argwhere((rs > 0.2) & (aprof == ringamax))
            print("RING irad ", irad, "rs[irad]",rs[irad])

        if 'amax' in file_input_profile:
            gapamax = np.min(aprof[(rs > 0.2)& (rs < 0.4)])
            irad = np.argwhere((rs > 0.2) & (aprof == gapamax) & (rs < 0.4))
            print("gapamax", gapamax, "GAP irad ", irad, "rs[irad]",rs[irad])


    if a_min is None:

        print("setting xrange from ", outputdir+modelimage)
        
        mrs, maprof, smaprof, dispmaprof, r1, r2 = getprofile(outputdir,
                                                              modelimage,
                                                              set_xrange=True)
        a_min = r1
        a_max = r2
        print("xrange 1 ", a_min)
        print("xrange 2 ", a_max)
    else:
        mrs, maprof, smaprof, dispmaprof = getprofile(outputdir,
                                                      modelimage,
                                                      set_xrange=False)

    if not ploterrors:
        dispmaprof = None

    if not drawlegend:
        linelabel = ''

    if file_input_profile:
        (y_min0, y_max0) = addprofile(axprofile,
                                      rs,
                                      a_min,
                                      a_max,
                                      aprof,
                                      dlo=None,
                                      dup=None,
                                      linelabel=linelabel,
                                      axislabel=axislabel,
                                      label='',
                                      linecolor='C0',
                                      y_min0=y_min0,
                                      y_max0=y_max0,
                                      visibleXaxis=True)
        linelabel = r'slab'

    if not drawlegend:
        linelabel = ''

    (y_min0, y_max0) = addprofile(axprofile,
                                  mrs,
                                  a_min,
                                  a_max,
                                  maprof,
                                  dlo=dispmaprof,
                                  dup=dispmaprof,
                                  linelabel=linelabel,
                                  axislabel=axislabel,
                                  label=label,
                                  linecolor='C1',
                                  y_min0=y_min0,
                                  y_max0=y_max0,
                                  visibleXaxis=True)

    return a_min, a_max, y_min0, y_max0


def exec_summary(datadir, outputdir='', UseInputDustImages=False, WithDustExpo=False, dofs=3):

    if WithDustExpo:
        nplotsx = 5
    else:
        nplotsx = 4
    nplotsy = 1
    figxsize = nplotsx*4
    figysize = 4.

    fig = plt.figure(constrained_layout=False, figsize=(figxsize, figysize))
    gs = fig.add_gridspec(
        nplotsy, nplotsx)  #, width_ratios=[2., 1., 1.], height_ratios=[1

    iplotpos=0
    axprofile_Tdust = fig.add_subplot(gs[0, iplotpos])
    
    if UseInputDustImages:
        file_input_profile=datadir + 'Tdust.fits'
    else:
        file_input_profile=datadir + 'Tdust_profile.dat'
        
    (a_min, a_max, y_min0,
     y_max0) = proc_1param(axprofile_Tdust,
                           file_input_profile=file_input_profile,
                           takelog=True,
                           axislabel='',
                           linelabel=r'$\log_{10}(T_{\rm d} \,/ \,\rm{K})$',
                           outputdir=outputdir,
                           modelimage='imlogTdust.fits',
                           label='a')
    
    if UseInputDustImages:
        file_input_profile=datadir + 'Sigma_g.fits'
    else:
        file_input_profile=datadir + 'Sigma_g_profile.dat'

    iplotpos+=1
    axprofile_Sigma_g = fig.add_subplot(gs[0, iplotpos])
    proc_1param(axprofile_Sigma_g,
                file_input_profile=file_input_profile,
                takelog=True,
                axislabel='',
                linelabel=r'$\log_{10}(\Sigma_{\rm g} \,/ \,\rm{g\,cm}^{-2})$',
                outputdir=outputdir,
                a_min=a_min,
                a_max=a_max,
                modelimage='imlogSigma_g.fits',
                label='b')

    if WithDustExpo:
        if UseInputDustImages:
            file_input_profile=datadir + 'qdustexpo.fits'
        else:
            file_input_profile=datadir + 'qdustexpo_profile.dat'
        
        iplotpos+=1
        axprofile_q_dustexpo = fig.add_subplot(gs[0, iplotpos])
        proc_1param(axprofile_q_dustexpo,
                    file_input_profile=file_input_profile,
                    takelog=False,
                    axislabel='',
                    linelabel=r'$q$',
                    outputdir=outputdir,
                    a_min=a_min,
                    a_max=a_max,
                    modelimage='imq_dustexpo.fits',
                    label='c')

    if UseInputDustImages:
        file_input_profile=datadir + 'amax.fits'
    else:
        file_input_profile=datadir + 'amax_profile.dat'

    iplotpos+=1
    axprofile_logamax = fig.add_subplot(gs[0, iplotpos])
    proc_1param(axprofile_logamax,
                file_input_profile=file_input_profile,
                takelog=True,
                axislabel='',
                linelabel=r'$\log_{10}(a_{\rm max} \,/ \,\rm{cm})$',
                outputdir=outputdir,
                a_min=a_min,
                a_max=a_max,
                modelimage='imlogamax.fits',
                label='c')

    iplotpos+=1
    axprofile_chi2 = fig.add_subplot(gs[0, iplotpos])
    proc_1param(axprofile_chi2,
                file_input_profile=False,
                takelog=True,
                axislabel='',
                linelabel=r'$\chi^2$,  ' + str(dofs) + 'dofs',
                outputdir=outputdir,
                a_min=a_min,
                a_max=a_max,
                modelimage='chi2map.fits',
                label='d')

    fileout_fig = outputdir + 'fig_profiles.pdf'
    plt.subplots_adjust(hspace=0.1)
    plt.subplots_adjust(wspace=0.25)
    print(fileout_fig)
    fig.savefig(fileout_fig, bbox_inches='tight')


def exec_stack(datadir, outputdirs=[], dofs=3, ploterrors=True):

    nplotsx = 5
    nplotsy = 1
    figxsize = 19.
    figysize = 4.

    fig = plt.figure(constrained_layout=False, figsize=(figxsize, figysize))
    gs = fig.add_gridspec(
        nplotsy, nplotsx)  #, width_ratios=[2., 1., 1.], height_ratios=[1

    axprofile_Tdust = fig.add_subplot(gs[0, 0])

    a_min = None
    a_max = None
    T_min0 = None
    T_max0 = None
    q_min0 = None
    q_max0 = None
    Sigmag_min0 = None
    Sigmag_max0 = None
    amax_min0 = None
    amax_max0 = None
    chi2_min0 = None
    chi2_max0 = None

    axprofile_Sigma_g = fig.add_subplot(gs[0, 1])
    axprofile_q_dustexpo = fig.add_subplot(gs[0, 2])
    axprofile_logamax = fig.add_subplot(gs[0, 3])
    axprofile_chi2 = fig.add_subplot(gs[0, 4])

    drawlegend = True
    for aoutputdir in outputdirs:

        (a_min, a_max, T_min0, T_max0) = proc_1param(
            axprofile_Tdust,
            file_input_profile=datadir + 'Tdust_profile.dat',
            takelog=True,
            axislabel='',
            linelabel=r'$\log_{10}(T_{\rm d} \,/ \,\rm{K})$',
            outputdir=aoutputdir,
            ploterrors=ploterrors,
            a_min=a_min,
            a_max=a_max,
            y_min0=T_min0,
            y_max0=T_max0,
            drawlegend=drawlegend,
            modelimage='imlogTdust.fits',
            label='')

        (a_min, a_max, Sigmag_min0, Sigmag_max0) = proc_1param(
            axprofile_Sigma_g,
            file_input_profile=datadir + 'Sigma_g_profile.dat',
            takelog=True,
            axislabel='',
            linelabel=r'$\log_{10}(\Sigma_{\rm g} \,/ \,\rm{g\,cm}^{-2})$',
            outputdir=aoutputdir,
            ploterrors=ploterrors,
            a_min=a_min,
            a_max=a_max,
            y_min0=Sigmag_min0,
            y_max0=Sigmag_max0,
            drawlegend=drawlegend,
            modelimage='imlogSigma_g.fits',
            label='')

        (a_min, a_max, q_min0, q_max0) = proc_1param(
            axprofile_q_dustexpo,
            file_input_profile=datadir + 'qdustexpo_profile.dat',
            takelog=False,
            axislabel='',
            linelabel=r'$q$',
            outputdir=aoutputdir,
            ploterrors=ploterrors,
            a_min=a_min,
            a_max=a_max,
            y_min0=q_min0,
            y_max0=q_max0,
            drawlegend=drawlegend,
            modelimage='imq_dustexpo.fits',
            label='')

        (a_min, a_max, amax_min0, amax_max0) = proc_1param(
            axprofile_logamax,
            file_input_profile=datadir + 'amax_profile.dat',
            takelog=True,
            axislabel='',
            linelabel=r'$\log_{10}(a_{\rm max} \,/ \,\rm{cm})$',
            outputdir=aoutputdir,
            ploterrors=ploterrors,
            a_min=a_min,
            a_max=a_max,
            y_min0=amax_min0,
            y_max0=amax_max0,
            drawlegend=drawlegend,
            modelimage='imlogamax.fits',
            label='')

        (a_min, a_max, chi2_min0,
         chi2_max0) = proc_1param(axprofile_chi2,
                                  file_input_profile=False,
                                  takelog=True,
                                  axislabel='',
                                  linelabel=r'$\chi^2$, expect ' + str(dofs),
                                  outputdir=aoutputdir,
                                  ploterrors=ploterrors,
                                  a_min=a_min,
                                  a_max=a_max,
                                  y_min0=chi2_min0,
                                  y_max0=chi2_max0,
                                  drawlegend=drawlegend,
                                  modelimage='chi2map.fits',
                                  label='')
        drawlegend = False

    fileout_fig = outputdirs[0] + 'fig_profiles_stacked.pdf'
    plt.subplots_adjust(hspace=0.1)
    plt.subplots_adjust(wspace=0.3)
    print(fileout_fig)
    fig.savefig(fileout_fig, bbox_inches='tight')
