import os
import sys
import re
from astropy.io import fits
import scipy
import scipy.signal

import matplotlib
from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import medfilt2d

include_path = '/home/simon/common/python/include/'
sys.path.append(include_path)

#from PyVtools import Vtools


def colorbar(Mappable, Orientation='horizontal', cbfmt="%.1e"):
    Ax = Mappable.axes
    fig = Ax.figure
    divider = make_axes_locatable(Ax)
    Cax = divider.append_axes("top", size="5%", pad=0.35)
    return fig.colorbar(mappable=Mappable,
                        cax=Cax,
                        use_gridspec=True,
                        orientation=Orientation,
                        format=cbfmt)


#def colorbar(Mappable, Orientation='horizontal'):
#    Ax = Mappable.axes
#    fig = Ax.figure
#    divider = make_axes_locatable(Ax)
#    Cax = divider.append_axes("top", size="5%", pad=0.35)
#    return fig.colorbar(
#            mappable=Mappable,
#            cax=Cax,
#            use_gridspec=True,
#            orientation=Orientation,
#            format="%.1e"
#    )
#


def addimage(
        iplotpos,
        label,
        atitle,
        filename_grey,
        filename_contours,
        #mask=None,
        errmask=None,
        #errthresh4mask=0.1,
        filename_serr=None,
        VisibleXaxis=False,
        VisibleYaxis=True,
        DoBeamEllipse=False,
        DoGreyCont=False,
        vsyst=0.,
        nplotsx=2,
        nplotsy=2,
        Range=False,
        SymmetricRange=False,
        MedianvalRange=False,
        DoCB=True,
        DoAxesLabels=True,
        cmap='RdBu_r',
        MedRms=True,
        Zoom=False,
        side=1.5,
        scaleunits=1.,
        LinearNotLog=False,
        ErrLinearNotLog=None,
        DoInterestingRegion=False,
        cbunits='Jy/beam',
        workdir='',
        cbfmt='%.2f'):

    ax = plt.subplot(nplotsy, nplotsx, iplotpos)
    # ax=axes[iplotpos]

    if DoAxesLabels:
        if ((iplotpos % nplotsx) == 1):
            ax.set_ylabel(r'$\delta$  offset / arcsec')
        if (iplotpos > (nplotsx * (nplotsy - 1))):
            ax.set_xlabel(r'$\alpha$ offset / arcsec')

    plt.setp(ax.get_xticklabels(), visible=VisibleXaxis)
    plt.setp(ax.get_yticklabels(), visible=VisibleYaxis)

    ax.tick_params(axis='both',
                   length=5,
                   width=1.,
                   color='grey',
                   direction='in',
                   left=True,
                   right=True,
                   bottom=True,
                   top=True)

    ax.spines['right'].set_color('grey')
    ax.spines['left'].set_color('grey')
    ax.spines['top'].set_color('grey')
    ax.spines['bottom'].set_color('grey')

    print("loading filename_grey", filename_grey)

    f = fits.open(filename_grey)
    im_grey = f[0].data * scaleunits
    hdr_grey = f[0].header
    cdelt = 3600. * hdr_grey['CDELT2']

    side0 = hdr_grey['NAXIS2'] * cdelt

    if Zoom:
        if (side > side0):
            sys.exit("side too large")

        nx = np.rint(side / cdelt)
        ny = np.rint(side / cdelt)

        i_star = np.rint(((0.) / hdr_grey['CDELT1']) +
                         (hdr_grey['CRPIX1'] - 1.))
        j_star = np.rint(((0.) / hdr_grey['CDELT2']) +
                         (hdr_grey['CRPIX2'] - 1.))

        j0 = int(j_star - (ny - 1.) / 2. + 1)
        j1 = int(j_star + (ny - 1.) / 2. + 1)
        i0 = int(i_star - (nx - 1.) / 2. + 1)
        i1 = int(i_star + (nx - 1.) / 2. + 1)
        subim_grey = im_grey[j0:j1, i0:i1]
        a0 = side / 2.
        a1 = -side / 2.
        d0 = -side / 2.
        d1 = side / 2.
    else:
        side = side0
        i0 = 0
        i1 = hdr_grey['NAXIS1']
        j0 = 0
        j1 = hdr_grey['NAXIS2']

        subim_grey = im_grey[:, :]
        a0 = (i0 - (hdr_grey['CRPIX1'] - 1)) * hdr_grey['CDELT1'] * 3600.
        a1 = (i1 - (hdr_grey['CRPIX1'] - 1)) * hdr_grey['CDELT1'] * 3600.
        d0 = (j0 - (hdr_grey['CRPIX2'] - 1)) * hdr_grey['CDELT2'] * 3600.
        d1 = (j1 - (hdr_grey['CRPIX2'] - 1)) * hdr_grey['CDELT2'] * 3600.

    if LinearNotLog & (ErrLinearNotLog is None):
        subim_grey = 10**(subim_grey)
    elif ErrLinearNotLog is not None:
        linear_expectval_subim = ErrLinearNotLog
        subim_grey = np.log(10) * linear_expectval_subim * subim_grey

    hdumask = fits.open(workdir + 'intensitymask.fits')
    #mask = np.ones(subim_grey.shape)
    intmask = hdumask[0].data
    subintmask = intmask[j0:j1, i0:i1]
    mask = (subintmask > 0)

    if errmask is not None:

        suberrmask = errmask[j0:j1, i0:i1]
        mask = ((suberrmask * subintmask) > 0)

    if MedianvalRange:
        typicalvalue = np.median(subim_grey[mask])
        rms = np.std(subim_grey[mask])
        medrms = np.sqrt(np.median((subim_grey[mask] - typicalvalue)**2))

        print("typical value ", typicalvalue, " rms ", rms, "medrms", medrms)
        range1 = np.min(subim_grey[mask])
        if MedRms:
            imagerms = medrms
        else:
            imagerms = rms
        range2 = typicalvalue + 3. * imagerms
        clevs = [range1, range2]
        clabels = ['%.1f' % (clevs[0]), '%.1f' % (clevs[1])]
    elif Range:
        range2 = Range[1]
        range1 = Range[0]
        clevs = [range1, range2]
        clabels = ['%.1f' % (range1), '%.1f' % (range2)]
    else:
        range2 = np.nanmax(subim_grey[mask])
        range1 = np.nanmin(subim_grey[mask])
        clevs = [range1, range2]
        clabels = ['%.1f' % (clevs[0]), '%.1f' % (clevs[1])]

    subim_grey[np.invert(mask)] = np.nan

    if ('sigma' in filename_grey):
        cmap = 'magma_r'

    #print("max:", np.max(subim_grey))
    #print("min:", np.min(subim_grey))
    #print("range1", range1, "range2", range2)
    if (np.isnan(subim_grey).any()):
        print("NaNs in subim_grey")
    #subim_grey = np.nan_to_num(subim_grey)

    #norm=LogNorm(vmin=range1, vmax=range2),

    theimage = ax.imshow(
        subim_grey,
        origin='lower',
        cmap=cmap,  #norm=norm,
        extent=[a0, a1, d0, d1],
        vmin=range1,
        vmax=range2,
        interpolation='nearest')  #'nearest'  'bicubic'

    ax.contour(
        mask,
        levels=[0.5],
        origin='lower',
        #color='black',
        #lw=10,
        extent=[a0, a1, d0, d1])  #,
    #interpolation='nearest')  #'nearest'  'bicubic'

    #plt.plot(0.,0.,marker='*',color='yellow',markersize=0.2,markeredgecolor='black')
    #plt.plot(0.,
    #         0.,
    #         marker='+',
    #         color='red',
    #         markersize=2.,
    #         alpha=0.6,
    #         lw=0.05)
    #
    #plt.plot(0.002,
    #         0.017,
    #         marker='+',
    #         color='green',
    #         markersize=2.,
    #         alpha=0.6,
    #         lw=0.05)

    ax.text(a0 * 0.9,
            d1 * 0.9,
            atitle,
            fontsize=12,
            ha='left',
            bbox=dict(facecolor='white', alpha=0.8))

    ax.text(a0 * 0.9,
            d0 * 0.9,
            label,
            weight='bold',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))

    axcb = plt.gca()

    if (DoCB):
        cb = colorbar(theimage, cbfmt=cbfmt)
        cb.ax.tick_params(labelsize='small')
        #print("CB label", cbunits)
        cb.set_label(cbunits)

    if DoBeamEllipse:
        from matplotlib.patches import Ellipse

        #Bmax/2 0.0579669470623286; Bmin/2 0.038567442164739;
        #PA-51.682370436407deg (South of East);

        bmaj = hdr_grey['BMAJ'] * 3600.
        bmin = hdr_grey['BMIN'] * 3600.
        bpa = hdr_grey['BPA']
        print("BEAM: %.3fx%.3f/%.0f" % (bmaj, bmin, bpa))
        e = Ellipse(xy=[a1 * 0.8, d0 * 0.8],
                    width=bmin,
                    height=bmaj,
                    angle=-bpa,
                    color='white')
        e.set_clip_box(axcb.bbox)
        e.set_facecolor('white')
        e.set_alpha(1.)
        axcb.add_artist(e)

    if DoInterestingRegion:
        #PA_to_HD147889 East of North -149.55224466034508
        x0 = 0.67
        y0 = 0.2
        PA = (360. - 149.552) * np.pi / 180.
        arrowlength = 0.45
        dx = np.sin(PA) * arrowlength
        dy = np.cos(PA) * arrowlength
        print("x0", x0, "a1", a0)
        ax.arrow(x0, y0, dx, dy, head_width=0.05, alpha=0.8)
        ax.text(x0 + dx - 0.05,
                y0 + dy - 0.05,
                'HD147889',
                fontsize=10,
                ha='left')

        #from matplotlib.patches import Ellipse
        ##Bmax/2 0.0579669470623286; Bmin/2 0.038567442164739;
        ##PA-51.682370436407deg (South of East);
        #bmaj = 0.25
        #bmin = 0.25
        #bpa = 0.
        #e = Ellipse(xy=[-0.276, -0.378],
        #            width=bmin,
        #            height=bmaj,
        #            angle=-bpa,
        #            color='yellow',
        #            fill=False)
        #e.set_clip_box(axcb.bbox)
        ##e.set_facecolor('yellow')
        ##e.set_alpha(0.5)
        #axcb.add_artist(e)

    return clevs, clabels, subim_grey


def exec_summary(workdir,
                 domain,
                 fileout,
                 DoCB=True,
                 DoAxesLabels=True,
                 WithAxes=True,
                 DoInterestingRegion=False,
                 LinearNotLog=False,
                 errthreshs=None,
                 RangeValues=None,
                 PlotFullDomain=False,
                 WithErrors=True,
                 Zoom=False,
                 side=1.2):
    """
    errthreshs = [['log(Tdust)', 0.2], ['log(amax)', 1.], ['log(Sigma_g)', 0.3]]
    defines the mask with thresholds for parameters in that list

    RangeValues={'log(amax)':[-0.58,2.0]},
    use to pass specific ranges rather than min/max into error threshold masks.

    """
    print("workdir:", workdir)
    #matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='sans-serif')
    #matplotlib.rcParams.update({'font.size': 16})
    font = {'family': 'Arial', 'weight': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    size_marker = 10

    # cmaps = ['magma', 'inferno', 'plasma', 'viridis', 'bone', 'afmhot', 'gist_heat', 'CMRmap', 'gnuplot', 'Blues_r', 'Purples_r', 'ocean', 'hot', 'seismic_r']
    gamma = 1.0

    nplotsy = 2
    nplotsx = int(len(domain))

    subfigsize = 3.
    figsize = (subfigsize * nplotsx, 1.2 * subfigsize * nplotsy)

    # (fig0, axes) = plt.subplots(nrows=nplotsy,ncols=nplotsx,figsize=figsize)

    plt.figure(figsize=figsize)

    iplotpos = 0

    cmap = 'inferno'
    #cmap = 'jet'
    cmap = 'hot'
    #cmap = 'ocean_r'
    labels = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n'
    ]
    VisibleXaxis = False
    VisibleYaxis = False

    #Range = False
    #Range = [1., 5.]
    DoInterestingRegion0 = False

    errmask = None

    from PyVtools import Vtools

    if errthreshs is not None:
        for apara in errthreshs:
            parname = apara[0]
            errthresh4mask = apara[1]
            if 'Tdust' in parname:
                rootname = 'imlogTdust.fits'
            elif 'amax' in parname:
                rootname = 'imlogamax.fits'
            elif 'expo' in parname:
                rootname = 'imq_dustexpo.fits'
            elif 'Sigma' in parname:
                rootname = 'imlogSigma_g.fits'
            else:
                sys.exit("no such parameter", parname)

            filename_grey = workdir + rootname
            filename_sup = workdir + 'sup' + rootname
            filename_sdo = workdir + 'sdo' + rootname
            filename_serr = workdir + 'serr' + rootname

            hduerr = fits.open(filename_serr)
            aim_err = hduerr[0].data
            hdr_err = hduerr[0].header
            amask = np.zeros(aim_err.shape, dtype='int')
            #amask = np.zeros(aim_err.shape)
            amask[(aim_err > 0) & (aim_err < errthresh4mask)] = 1
            if errmask is None:
                errmask = amask
            else:
                errmask = errmask * amask

    ThisLinearNotLog = LinearNotLog
    for ipara, apara in enumerate(domain):
        iplotpos += 1
        parname = apara[0]
        if 'Tdust' in parname:
            mask = None
            rootname = 'imlogTdust.fits'
            if ThisLinearNotLog:
                atitle = r'${\rm T}_{\rm d}/{\rm K}$'
            else:
                atitle = r'$\log({\rm T}_{\rm d}/{\rm K})$'
            cmap = 'hot'
            if DoInterestingRegion:
                DoInterestingRegion0 = True
        elif 'amax' in parname:
            ThisLinearNotLog = False
            rootname = 'imlogamax.fits'
            if ThisLinearNotLog:
                atitle = r'${\rm a}_{\rm max}/{\rm cm}$'
            else:
                atitle = r'$\log({\rm a}_{\rm max}/{\rm cm})$'
            #cmap = 'jet'
            cmap = 'Greens'
        elif 'expo' in parname:
            ThisLinearNotLog = False
            rootname = 'imq_dustexpo.fits'
            atitle = r'$q$'
            cmap = 'jet'
        elif 'Sigma' in parname:
            ThisLinearNotLog = False
            rootname = 'imlogSigma_g.fits'
            if ThisLinearNotLog:
                atitle = r'${\Sigma}_{\rm g}/{\rm g\,cm}^{-2}$'
            else:
                atitle = r'$\log({\Sigma}_{\rm g}/{\rm g\,cm}^{-2})$'
            cmap = 'Blues'
            #cmap = 'binary'
        else:
            sys.exit("no such parameter", parname)

        filename_grey = workdir + rootname
        filename_sup = workdir + 'sup' + rootname
        filename_sdo = workdir + 'sdo' + rootname
        filename_serr = workdir + 'serr' + rootname
        label = labels[ipara]
        cbunits = ''

        filename_contours = False

        if PlotFullDomain:
            passrangevalues = apara[1]
        elif RangeValues is not None:
            if parname in RangeValues.keys():
                passrangevalues = RangeValues[parname]
            else:
                passrangevalues = None
        else:
            passrangevalues = None
            
        if WithAxes:
            VisibleXaxis = True
            VisibleYaxis = False
            if WithErrors:
                VisibleXaxis = False
            if iplotpos == 1:
                VisibleYaxis = True

        (clevs, clabels, subim_grey) = addimage(
            iplotpos,
            label,
            atitle,
            filename_grey,
            errmask=errmask,
            filename_serr=filename_serr,
            filename_contours=filename_contours,
            VisibleXaxis=VisibleXaxis,
            VisibleYaxis=VisibleYaxis,
            DoBeamEllipse=True,
            DoGreyCont=False,
            nplotsx=nplotsx,
            nplotsy=nplotsy,
            SymmetricRange=False,
            DoCB=DoCB,
            DoAxesLabels=DoAxesLabels,
            cmap=cmap,
            Range=passrangevalues,
            Zoom=Zoom,
            side=side,
            DoInterestingRegion=DoInterestingRegion0,
            cbunits=cbunits,
            scaleunits=1.,
            workdir=workdir,
            LinearNotLog=ThisLinearNotLog,
            cbfmt='%.2f')
        DoInterestingRegion0 = False
        if WithErrors:
            ierrplotpos = iplotpos + nplotsx
            atitle = atitle + ' error'
            label = labels[ierrplotpos - 1]
            filename_grey = filename_serr
            cmap = 'binary'
            #cmap='Greys'
            if ThisLinearNotLog:
                ThisErrLinearNotLog = subim_grey
            else:
                ThisErrLinearNotLog = None

            (clevs, clabels, errsubim_grey) = addimage(
                ierrplotpos,
                label,
                atitle,
                filename_grey,
                errmask=errmask,
                workdir=workdir,
                filename_contours=filename_contours,
                VisibleXaxis=True,
                VisibleYaxis=VisibleYaxis,
                DoBeamEllipse=True,
                DoGreyCont=False,
                nplotsx=nplotsx,
                #Range=Range,
                nplotsy=nplotsy,
                SymmetricRange=False,
                DoCB=DoCB,
                DoAxesLabels=DoAxesLabels,
                cmap=cmap,
                Zoom=Zoom,
                #Range=clevs,
                side=side,
                LinearNotLog=ThisLinearNotLog,
                ErrLinearNotLog=ThisErrLinearNotLog,
                DoInterestingRegion=False,
                cbunits=cbunits,
                cbfmt='%.2f')

    plt.subplots_adjust(hspace=0.05)
    #plt.subplots_adjust(wspace=0.1)

    #plt.subplots_adjust(hspace=0.)
    plt.subplots_adjust(wspace=0.05)

    print(fileout)
    plt.tight_layout()

    #plt.savefig(fileout, bbox_inches='tight', dpi=500)
    plt.savefig(fileout, bbox_inches='tight')

    #plt.savefig(fileout)

    return
