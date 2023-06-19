import os
import re
from astropy.io import fits
import scipy
import scipy.signal

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import medfilt2d

include_path = '/Users/simon/common/python/include/'
sys.path.append(include_path)

#import Vtools


def colorbar(Mappable, Orientation='horizontal', cbfmt="%.1e"):
    Ax = Mappable.axes
    fig = Ax.figure
    divider = make_axes_locatable(Ax)
    Cax = divider.append_axes("top", size="5%", pad=0.55)
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


def addimage(iplotpos,
             label,
             atitle,
             filename_grey,
             filename_contours,
             VisibleXaxis=False,
             VisibleYaxis=True,
             DoBeamEllipse=False,
             DoGreyCont=False,
             vsyst=0.,
             nplotsx=2,
             nplotsy=2,
             SymmetricRange=False,
             MedianvalRange=False,
             DoCB=True,
             cmap='RdBu_r',
             MedRms=True,
             Zoom=False,
             side=1.5,
             scaleunits=1E3,
             DoInterestingRegion=False,
             cbunits='Jy/beam',
             cbfmt='%.2f'):

    print("nplotsx ", nplotsx, iplotpos)
    ax = plt.subplot(nplotsy, nplotsx, iplotpos)
    # ax=axes[iplotpos]

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

    if ((iplotpos % nplotsx) == 1):
        ax.set_ylabel(r'$\delta$  offset / arcsec')
    if (iplotpos > (nplotsx * (nplotsy - 1))):
        ax.set_xlabel(r'$\alpha$ offset / arcsec')

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

    else:
        side = side0
        i0 = 0
        i1 = hdr_grey['NAXIS1'] - 1
        j0 = 0
        j1 = hdr_grey['NAXIS2'] - 1

        #subim_grey=im_grey.copy()
        subim_grey = im_grey[ :, :]

    a0 = side / 2.
    a1 = -side / 2.
    d0 = -side / 2.
    d1 = side / 2.

    # if 'v' in filename_grey:
    #	subim_grey = subim_grey - vsyst

    mask = np.ones(subim_grey.shape)
    mask = np.where(mask > 0)

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
    else:
        range2 = np.max(subim_grey[mask])
        range1 = np.min(subim_grey[mask])
        clevs = [range1, range2]
        clabels = ['%.1f' % (clevs[0]), '%.1f' % (clevs[1])]

    if ('sigma' in filename_grey):
        cmap = 'magma_r'

    print("max:", np.max(subim_grey))
    print("min:", np.min(subim_grey))
    print("range1", range1, "range2", range2)
    if (np.isnan(subim_grey).any()):
        print("NaNs in subim_grey")
    subim_grey = np.nan_to_num(subim_grey)

    theimage = ax.imshow(
        subim_grey,
        origin='lower',
        cmap=cmap,  #norm=norm,
        extent=[a0, a1, d0, d1],
        vmin=range1,
        vmax=range2,
        interpolation='nearest')  #'nearest'  'bicubic'

    #plt.plot(0.,0.,marker='*',color='yellow',markersize=0.2,markeredgecolor='black')
    plt.plot(0.,
             0.,
             marker='+',
             color='red',
             markersize=2.,
             alpha=0.6,
             lw=0.05)

    plt.plot(0.002,
             0.017,
             marker='+',
             color='green',
             markersize=2.,
             alpha=0.6,
             lw=0.05)

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
        print("CB label", cbunits)
        cb.set_label(cbunits)

        #cmap1 = cmap
        #norm = mpl.colors.Normalize(vmin=range1, vmax=range2)
        #fig=plt.gcf()
        #cbar_ax = fig.add_axes([0.92, 0.62, 0.01, 0.15])
        #if (iplotpos > 2):
        #        cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.15])
        #
        #print( "color bar clevs",clevs)
        #print( "color bar clabels",clabels)
        #
        #
        #cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=colors.Colormap(cmap1), norm=norm, orientation='vertical', ticks=clevs)
        #
        #cb.ax.set_yticklabels(clabels)
        ## cb.ax.set_ylabel('km/s', rotation=270)
        #xcb.ax.tick_params(labelsize=12)

    # if (isinstance(DoGreyCont,str)):
    #         filename_grey_cont=DoGreyCont
    #
    #
    #
    #         f_cont = fits.open(filename_grey_cont)
    #         im_grey_cont = f_cont[0].data
    #         hdr_grey_cont= f_cont[0].header
    #         subim_grey_cont = im_grey_cont[int(j0):int(j1),int(i0):int(i1)]
    #         print( "i0 "+str(i0),"hdr_grey['CRPIX2']", hdr_grey['CRPIX2'])
    #         levels = [-1.65344452058093, -1.6129840879707 , -1.55158301988206, -1.51002707264227   ]
    #
    #
    #         CS = axcb.contour(subim_grey_cont,levels , origin='lower', linewidths=1.0,
    #                           linestyles = 'solid',
    #                           extent=[a0,a1,d0,d1], colors='red')
    #
    # elif (DoGreyCont):
    #        levels=np.array((vsyst))
    #        CS = axcb.contour(subim_grey,levels , origin='lower', linewidths=0.5,
    #                          linestyles = 'solid',
    #                          extent=[a0,a1,d0,d1], colors='green')

    #if (filename_contours!=False):
    #
    #        ######################################################################
    #        #  contours
    #
    #	print( "loading filename_contours",filename_contours)
    #        f = fits.open(filename_contours)
    #        im_cont = f[0].data
    #        hdr_cont= f[0].header
    #
    #        subim_cont = im_cont[int(j0):int(j1),int(i0):int(i1)]
    #
    #        if MaskCont:
    #                subim_cont[np.where(subim_region < 0.5)] = 0.
    #
    #
    #
    #        ######################################################################
    #        #Blue
    #        print( "full dvel range: ", np.min(subim_cont), " --> ",np.max(subim_cont))
    #        levels=np.array([0.6,])*np.fabs(np.min(subim_cont))
    #        #levels=np.array([0.7421825906386711,])
    #        alphas=np.ones(len(levels)) #  0.5+0.5*np.arange(len(levels))/(len(levels))
    #        linewidths=np.ones(len(levels)) # 1.0-((0.+np.arange(len(levels))-1.0)/(len(levels))) *0.5
    #
    #        levels_list = levels.tolist()
    #        for ilevel in range(len(levels)):
    #                alpha_val = alphas[ilevel]
    #                alinew=linewidths[ilevel]
    #                alevel = -levels[ilevel]
    #                print( "blue level",alevel,"alpha_val",alpha_val,"lw",alinew)
    #                CS = axcb.contour(subim_cont, alevel, origin='lower', linewidths=alinew,
    #                                  linestyles = 'solid',
    #                                  extent=[a0,a1,d0,d1], colors='blue', alpha = alpha_val  )
    #
    #        ######################################################################
    #        #Red
    #        #levels=np.array([0.75,0.95])
    #
    #
    #        levels=np.array([0.6,])*np.fabs(np.max(subim_cont))
    #        #levels=np.array([0.6441364043826032,])
    #        levels_list = levels.tolist()
    #        for ilevel in range(len(levels)):
    #                alpha_val = alphas[ilevel]
    #                alinew=linewidths[ilevel]
    #                alevel = levels[ilevel]
    #                print( "red level",alevel,"alpha_val",alpha_val,"lw",alinew)
    #                CS = axcb.contour(subim_cont, alevel, origin='lower', linewidths=alinew,
    #                                  linestyles = 'solid',
    #                                  extent=[a0,a1,d0,d1], colors='red', alpha = alpha_val  )
    #
    #
    #

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
                    color='blue')
        e.set_clip_box(axcb.bbox)
        e.set_facecolor('yellow')
        e.set_alpha(0.5)
        axcb.add_artist(e)

    if DoInterestingRegion:
        from matplotlib.patches import Ellipse

        #Bmax/2 0.0579669470623286; Bmin/2 0.038567442164739;
        #PA-51.682370436407deg (South of East);

        bmaj = 0.25
        bmin = 0.25
        bpa = 0.
        e = Ellipse(xy=[-0.276, -0.378],
                    width=bmin,
                    height=bmaj,
                    angle=-bpa,
                    color='yellow',
                    fill=False)
        e.set_clip_box(axcb.bbox)
        #e.set_facecolor('yellow')
        #e.set_alpha(0.5)
        axcb.add_artist(e)

    return clevs, clabels


def exec_summary(workdir, fileout, Zoom=False, side=1.2):

    # global nplotsx
    # global nplotsy
    # global basename_log

    print("workdir:", workdir)
    #matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='sans-serif')
    #matplotlib.rcParams.update({'font.size': 16})
    font = {'family': 'Arial', 'weight': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    size_marker = 10

    # cmaps = ['magma', 'inferno', 'plasma', 'viridis', 'bone', 'afmhot', 'gist_heat', 'CMRmap', 'gnuplot', 'Blues_r', 'Purples_r', 'ocean', 'hot', 'seismic_r']
    gamma = 1.0

    figsize = (18., 5.)
    nplotsx = 4
    nplotsy = 1

    # (fig0, axes) = plt.subplots(nrows=nplotsy,ncols=nplotsx,figsize=figsize)

    plt.figure(figsize=figsize)
    #axes=axes.flatten()
    #print("axes",help(axes))
    #print((axes.shape))

    iplotpos = 0

    cmap = 'inferno'
    atitle = r'$100\,$GHz'
    label = 'a'
    filename_contours = False
    filename_grey = workdir + 'I_100.fits'
    iplotpos += 1
    #addimage(iplotpos,label,atitle,filename_grey,filename_contours,VisibleXaxis=False,VisibleYaxis=True,DoBeamEllipse=False,DoGreyCont=True,Clevs=[vsyst-vrange,vsyst,vsyst+vrange])
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_contours=filename_contours,
                                VisibleXaxis=True,
                                VisibleYaxis=True,
                                DoBeamEllipse=True,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=Zoom,
                                side=side,
                                DoInterestingRegion=False,
                                cbunits='mJy/beam',
                                cbfmt='%.1f')

    atitle = r'$150\,$GHz'
    label = 'b'
    filename_contours = False
    filename_grey = workdir + 'I_150.fits'
    iplotpos += 1
    #addimage(iplotpos,label,atitle,filename_grey,filename_contours,VisibleXaxis=False,VisibleYaxis=True,DoBeamEllipse=False,DoGreyCont=True,Clevs=[vsyst-vrange,vsyst,vsyst+vrange])
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_contours=filename_contours,
                                VisibleXaxis=True,
                                VisibleYaxis=True,
                                DoBeamEllipse=True,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=Zoom,
                                side=side,
                                DoInterestingRegion=False,
                                cbunits='mJy/beam',
                                cbfmt='%.1f')

    atitle = r'$230\,$GHz'
    label = 'c'
    filename_contours = False
    filename_grey = workdir + 'I_230.fits'
    iplotpos += 1
    #addimage(iplotpos,label,atitle,filename_grey,filename_contours,VisibleXaxis=False,VisibleYaxis=True,DoBeamEllipse=False,DoGreyCont=True,Clevs=[vsyst-vrange,vsyst,vsyst+vrange])
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_contours=filename_contours,
                                VisibleXaxis=True,
                                VisibleYaxis=True,
                                DoBeamEllipse=True,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=Zoom,
                                side=side,
                                DoInterestingRegion=False,
                                cbunits='mJy/beam',
                                cbfmt='%.1f')
    
    atitle = r'$345\,$GHz'
    label = 'd'
    filename_contours = False
    filename_grey = workdir + 'I_345.fits'
    iplotpos += 1
    #addimage(iplotpos,label,atitle,filename_grey,filename_contours,VisibleXaxis=False,VisibleYaxis=True,DoBeamEllipse=False,DoGreyCont=True,Clevs=[vsyst-vrange,vsyst,vsyst+vrange])
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_contours=filename_contours,
                                VisibleXaxis=True,
                                VisibleYaxis=True,
                                DoBeamEllipse=True,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=Zoom,
                                side=side,
                                DoInterestingRegion=False,
                                cbunits='mJy/beam',
                                cbfmt='%.1f')


    #plt.subplots_adjust(hspace=0.)
    #plt.subplots_adjust(wspace=0.)

    print(fileout)
    #plt.tight_layout()

    plt.savefig(fileout, bbox_inches='tight', dpi=500)

    #plt.savefig(fileout)

    return


workdir = './data/'

fileout = workdir + 'fig_summary_mock.png'

exec_summary(workdir, fileout, Zoom=True, side=1.9)
