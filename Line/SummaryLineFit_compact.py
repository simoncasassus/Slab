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
from  scipy.signal import medfilt2d

include_path='/home/simon/common/python/include/'
sys.path.append(include_path)
import ImUtils.Resamp as Resamp
import ImUtils.Cube2Im as Cube2Im

#import Vtools

def colorbar(Mappable, Orientation='horizontal',cbfmt="%.1e"):
    Ax = Mappable.axes
    fig = Ax.figure
    divider = make_axes_locatable(Ax)
    Cax = divider.append_axes("top", size="5%", pad=0.55)
    return fig.colorbar(
            mappable=Mappable,
            cax=Cax,
            use_gridspec=True,
            orientation=Orientation,
            format=cbfmt
    )

def addimage(iplotpos,label,atitle,filename_grey,filename_contours,filename_errormap=False, filename_fiterrormap=False,VisibleXaxis=False,VisibleYaxis=True,DoBeamEllipse=False,DoGreyCont=False,vsyst=0.,nplotsx=2,nplotsy=2,SymmetricRange=False,MedianvalRange=False,DoCB=True,cmap='RdBu_r',MedRms=True,Zoom=True,scaleim=1.,cbfmt="%.1e",cbunits='',DoInterestingRegion=False,side=1.2,Range=False):

        print( "nplotsy nplotsx ", nplotsy, nplotsx, iplotpos)
        ax = plt.subplot(nplotsy, nplotsx, iplotpos)
        # ax=axes[iplotpos]

        plt.setp(ax.get_xticklabels(),visible=VisibleXaxis)
        plt.setp(ax.get_yticklabels(),visible=VisibleYaxis)

        ax.tick_params(axis='both',length = 5, width=1., color = 'grey',direction='in',left=True, right=True,bottom=True, top=True)

        ax.spines['right'].set_color('grey')
        ax.spines['left'].set_color('grey')
        ax.spines['top'].set_color('grey')
        ax.spines['bottom'].set_color('grey')




        if ((iplotpos % nplotsx) == 1):
                ax.set_ylabel(r'$\delta$  offset / arcsec')
        #if (iplotpos > (nplotsx*(nplotsy-1))):
        #        # ax.set_xlabel(r'$\alpha$ offset / arcsec')


        print( "loading filename_grey",filename_grey)

        f = fits.open(filename_grey)
        f=Cube2Im.slice0(f)
        im_grey = f.data
        hdr_grey= f.header
        cdelt=3600.*hdr_grey['CDELT2']

        side0=hdr_grey['NAXIS2']*cdelt


        if Zoom:
                
                if (side > side0):
                        sys.exit("side too large")



                nx=np.rint(side/cdelt)
                ny=np.rint(side/cdelt)

                Resample=True
                if Resample:

                        hdrzoom=hdr_grey.copy()
                        hdrzoom['NAXIS1']=nx
                        hdrzoom['NAXIS2']=ny
                        hdrzoom['CRPIX1']=((nx-1.)/2.)+1.
                        hdrzoom['CRPIX2']=((ny-1.)/2.)+1.

                        hduzoom=Resamp.gridding(f,hdrzoom,ReturnHDU=True)
                        subim_grey=hduzoom.data
                        hdr_grey=hduzoom.header

                else:
                        i_star = hdr_grey['CRPIX1']-1.
                        j_star = hdr_grey['CRPIX2']-1.

                        i0=int(i_star-(nx-1.)/2.+0.5)
                        j0=int(j_star-(ny-1.)/2.+0.5)
                        i1=int(i_star+(nx-1.)/2.+0.5)
                        j1=int(j_star+(ny-1.)/2.+0.5)


                        subim_grey = im_grey[j0:j1,i0:i1]



                        #j0=int(j_star-(ny-1.)/2.+1)
                        #j1=int(j_star+(ny-1.)/2.+1)
                        #i0=int(i_star-(nx-1.)/2.+1)
                        #i1=int(i_star+(nx-1.)/2.+1)

        else:
                side=side0
                i0=0
                i1=hdr_grey['NAXIS1']-1
                j0=0
                j1=hdr_grey['NAXIS2']-1

                subim_grey=im_grey.copy()


        a0 = side/2.
        a1 = -side/2.
        d0 = -side/2.
        d1 = side/2.

        subim_grey *= scaleim

        # if 'v' in filename_grey:
        #	subim_grey = subim_grey - vsyst


        if (filename_fiterrormap):
            print( "loading filename_fiterrormap",filename_fiterrormap)
            
            f = fits.open(filename_fiterrormap)
            im_fiterrormap = f[0].data
            hdr_fiterrormap= f[0].header
            if Zoom:
                subim_fiterrormap=im_fiterrormap[j0:j1,i0:i1]
            else:
                subim_fiterrormap=im_fiterrormap.copy()
                

            typicalerror=np.median(subim_fiterrormap)

            mask=np.where(subim_fiterrormap<3.*typicalerror)

            immask=np.zeros(subim_fiterrormap.shape)
            immask[mask]=1
            print("number of pixels masked:",np.sum(immask))

        else:
            mask=np.where((isfinite(subim_grey)) & (subim_grey > 0.)  & (subim_grey > np.min(subim_grey)))
            
        # print("viewing fiterrormap immask")
        # Vtools.View(immask)


        # plt.figure(2)
        # plt.imshow(immask)
        # plt.show()
        # plt.figure(1)

        if filename_errormap: 
                f = fits.open(filename_errormap)
                im_errormap = f[0].data
                hdr_errormap= f[0].header

                if Zoom:
                        subim_errormap=im_errormap[j0:j1,i0:i1]
                else:
                        subim_errormap=im_errormap.copy()

                medsubim_errormap=medfilt2d(subim_errormap,kernel_size=5)
                mask=np.where(mask and ( (subim_errormap-medsubim_errormap) > 3.* medsubim_errormap))


                immask=np.zeros(subim_fiterrormap.shape)
                immask[mask]=1.
                print("viewing errormap immask")
                #Vtools.View(immask)

        if Range:
            range1=Range[0]
            range2=Range[1]
            clevs = [range1,range2]
            clabels=['%.1f' % (clevs[0]),'%.1f' % (clevs[1])]
        elif SymmetricRange:
            range2=vsyst+SymmetricRange
            range1=vsyst-SymmetricRange
            subim_grey[(np.fabs(subim_grey) == 0.)]=vsyst 
            # subim_grey[np.where(subim_grey < range1)]=vsyst
            # subim_grey[np.where(subim_grey > range2)]=vsyst
            clevs = [range1,0.,range2]
            clabels=['%.0f' % (clevs[0]),'','%.0f' % (clevs[2])]
        elif MedianvalRange:
                typicalvalue=np.median(subim_grey[mask])

                rms=np.std(subim_grey[mask])

                medrms=np.sqrt(np.median( (subim_grey[mask] - typicalvalue)**2))

                print("MedianvalRange typical value ",typicalvalue," rms ",rms," medrms ",medrms)
                
                range1=np.min(subim_grey[mask])
                if MedRms:
                        imagerms=medrms
                else:
                        imagerms=rms
                range2=typicalvalue+MedianvalRange*imagerms
                clevs = [range1,range2]
                clabels=['%.1f' % (clevs[0]),'%.1f' % (clevs[1])]
                print("range:  ",clevs)
        else:
                range2=np.max(subim_grey[mask])
                range1=np.min(subim_grey[mask])
                clevs = [range1,range2]
                clabels=['%.1f' % (clevs[0]),'%.1f' % (clevs[1])]


        #if ('sigma' in filename_grey):
        #        cmap='inferno'

        print("max:",np.max(subim_grey))
        print("min:",np.min(subim_grey))
        print("range1",range1,"range2",range2)
        if (np.isnan(subim_grey).any()):
                print("NaNs in subim_grey")
        subim_grey=np.nan_to_num(subim_grey)


        theimage=ax.imshow(subim_grey, origin='lower', cmap=cmap, #norm=norm,
                   extent=[a0,a1,d0,d1], vmin=range1, vmax=range2, interpolation='nearest') #'nearest'  'bicubic'

        #plt.plot(0.,0.,marker='*',color='yellow',markersize=0.2,markeredgecolor='black')
        plt.plot(0.,0.,marker='*',color='yellow',markersize=0.4)


        ax.text(a1*0.9,d0*0.9,atitle,weight='bold',fontsize=14,ha='right',bbox=dict(facecolor='white', alpha=0.8))

        ax.text(a0*0.9,d0*0.9,label,weight='bold',fontsize=12,bbox=dict(facecolor='white', alpha=0.8))

        axcb=plt.gca()

        if (DoCB):
                cb=colorbar(theimage,cbfmt=cbfmt)
                cb.ax.tick_params(labelsize='small')
                print("CB label",cbunits)
                cb.set_label(cbunits)


        if (filename_contours!=False):
            
            ######################################################################
            #  contours
                
            print( "loading filename_contours",filename_contours)
            hducontours = fits.open(filename_contours)
            hdr_cont = hducontours[0].header
            hdr_cont['CRVAL1']=hdr_grey['CRVAL1']
            hdr_cont['CRVAL2']=hdr_grey['CRVAL2']
            hducontours[0].header=hdr_cont
            

            hducontours_matched=Resamp.gridding(hducontours,hdr_grey,ReturnHDU=True)
            
            subim_contours = hducontours_matched.data
            
            levels=np.array([0.2,0.3,0.4,0.6,0.8])*np.max(subim_contours)
            ax.contour(subim_contours,origin='lower', cmap='magma_r', extent=[a0,a1,d0,d1], levels=levels,linewidths=1.0,alpha=0.5)

            ax.imshow(subim_contours,origin='lower', cmap='magma_r', extent=[a0,a1,d0,d1], alpha=0.4,vmax=0.5,vmin=0.)


            hducontours_matched.writeto('matched_contours.fits',overwrite=True)
        

        if DoBeamEllipse:
                from matplotlib.patches import Ellipse

                #Bmax/2 0.0579669470623286; Bmin/2 0.038567442164739;
                #PA-51.682370436407deg (South of East);

                bmaj = hdr_grey['BMAJ'] * 3600.
                bmin = hdr_grey['BMIN'] * 3600.
                bpa = hdr_grey['BPA']
                e = Ellipse(xy=[a1*0.8,d1*0.8], width=bmin, height=bmaj, angle=-bpa,color='blue')
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
                e = Ellipse(xy=[-0.276,-0.378], width=bmin, height=bmaj, angle=-bpa,color='yellow',fill=False)
                e.set_clip_box(axcb.bbox)
                #e.set_facecolor('yellow')
                #e.set_alpha(0.5)
                axcb.add_artist(e)





        return  clevs, clabels




    
                

def exec_summary(workdir,fileout,vsyst=0.,vrange=10.,fix_vturb=False, WCont=True,Zoom=False,RunMCMC=False,velorange=4.,side=1.2,fix_T=False,SigmaScale=[1.,''],cmap_Sigmag='gist_earth_r'):

        # global nplotsx
        # global nplotsy
        # global basename_log


        print( "workdir:",workdir)
        if (not re.search(r"\/$",workdir)):
            workdir+='/'
            print("added trailing back slack to workdir")

        #matplotlib.rc('text', usetex=True) 
        matplotlib.rc('font', family='sans-serif') 
        #matplotlib.rcParams.update({'font.size': 16})
        font = {'family' : 'Arial',
                'weight' : 'normal',
                'size'   : 12}

        matplotlib.rc('font', **font)


        size_marker=10

        # cmaps = ['magma', 'inferno', 'plasma', 'viridis', 'bone', 'afmhot', 'gist_heat', 'CMRmap', 'gnuplot', 'Blues_r', 'Purples_r', 'ocean', 'hot', 'seismic_r']
        gamma=1.0

        if ( ('fix_t' in workdir.lower()) or   ('fixt' in workdir.lower()) ):
            fix_T=True

        if ( ('fix_vturb' in workdir.lower()) or   ('fixvturb' in workdir.lower()) ):
            fix_vturb=True

        if (fix_vturb or fix_T):
            figsize=(12.,5.)
            nplotsx=3
            nplotsy=1
        else:
            figsize=(15., 5.)
            nplotsx=4
            nplotsy=1

        if os.path.isfile(workdir+'errtemperature.fits'):
            RunMCMC=True
            
        if RunMCMC:
            if (fix_vturb or fix_T):
                figsize=(12.,10.)
                nplotsx=3
                nplotsy=2
            else:
                figsize=(15.,10.)
                nplotsx=4
                nplotsy=2

        labels=['a','b','c','d','e','f','g','h']
        
        # (fig0, axes) = plt.subplots(nrows=nplotsy,ncols=nplotsx,figsize=figsize)

        plt.figure(figsize=figsize)
        #axes=axes.flatten()
        #print("axes",help(axes))
        #print((axes.shape))

        iplotpos=0

        filename_fiterrormap=workdir+'fiterror.fits'
        filename_fiterrormap=False

        cmap='ocean_r'
        #cmap='magma_r'
        
        #cmap_Sigmag='RdGy_r'
        #cmap_Sigmag='ocean_r'
        cmap_vturb='gist_yarg'
        ilabel=0
        atitle=r'$\Sigma_g$'
        label=labels[ilabel] 
        filename_contours=False
        filename_grey=workdir+'Sigma_g.fits'
        filename_errormap=False
        iplotpos += 1
        #addimage(iplotpos,label,atitle,filename_grey,filename_contours,VisibleXaxis=False,VisibleYaxis=True,DoBeamEllipse=False,DoGreyCont=True,Clevs=[vsyst-vrange,vsyst,vsyst+vrange])
        (clevs,clabels)=addimage(iplotpos,label,atitle,filename_grey,filename_contours=filename_contours,filename_errormap=filename_errormap, filename_fiterrormap=filename_fiterrormap,VisibleXaxis=True,VisibleYaxis=True,DoBeamEllipse=True,DoGreyCont=False,vsyst=vsyst,nplotsx=nplotsx,nplotsy=nplotsy,SymmetricRange=False,MedianvalRange=False,DoCB=True, cmap=cmap_Sigmag,cbfmt='%.2f',cbunits=r' '+SigmaScale[1]+r'$\rm{g}\,\rm{cm}^{-2}$',Zoom=Zoom,DoInterestingRegion=False,side=side,scaleim=SigmaScale[0])
        ilabel+=1

        if not fix_T:
            atitle=r'$T$'
            label=labels[ilabel] 
            ilabel+=1
            filename_contours=False
            filename_grey=workdir+'temperature.fits'
            filename_errormap=False
            iplotpos += 1
            #addimage(iplotpos,label,atitle,filename_grey,filename_contours,VisibleXaxis=False,VisibleYaxis=False,DoBeamEllipse=False,DoGreyCont=True,Clevs=[vsyst-vrange,vsyst,vsyst+vrange])
            (clevs,clabels)=addimage(iplotpos,label,atitle,filename_grey,filename_contours=filename_contours,filename_errormap=filename_errormap, filename_fiterrormap=filename_fiterrormap,VisibleXaxis=True,VisibleYaxis=False,DoBeamEllipse=True,DoGreyCont=False,vsyst=vsyst,nplotsx=nplotsx,nplotsy=nplotsy,SymmetricRange=False,MedianvalRange=False,DoCB=True, cmap='YlOrBr',scaleim=1.,cbfmt='%.1f',cbunits=r'$\rm{K}$',Zoom=Zoom,DoInterestingRegion=False,side=side,Range=[0.,130.])


        atitle=r'$v^\circ$'
        label=labels[ilabel]
        ilabel+=1
        filename_contours=False
        filename_grey=workdir+'velocentroid.fits'
        filename_errormap=False
        iplotpos += 1
        #addimage(iplotpos,label,atitle,filename_grey,filename_contours,VisibleXaxis=False,VisibleYaxis=False,DoBeamEllipse=False,DoGreyCont=True,Clevs=[vsyst-vrange,vsyst,vsyst+vrange])
        (clevs,clabels)=addimage(iplotpos,label,atitle,filename_grey,filename_contours=filename_contours,filename_errormap=filename_errormap, filename_fiterrormap=filename_fiterrormap,VisibleXaxis=True,VisibleYaxis=False,DoBeamEllipse=True,DoGreyCont=False,vsyst=vsyst,nplotsx=nplotsx,nplotsy=nplotsy,SymmetricRange=velorange,DoCB=True, cmap='RdBu_r',scaleim=1.,cbfmt='%.1f',cbunits=r'$\rm{km}\,\rm{s}^{-1}$',Zoom=Zoom,DoInterestingRegion=False,side=side)
                

        

        if (not fix_vturb):
            atitle=r'$v_\mathrm{turb}$'
            label=labels[ilabel]
            ilabel+=1
            filename_contours='/home/simon/common/ppdisks/HD135344B/data/SPHERE/IRDAP/HD135344B_2016-06-30_Q_phi_star_pol_subtr_whdr.fits'
            filename_contours=False
            filename_grey=workdir+'vturb.fits'
            filename_errormap=False
            iplotpos += 1
            #addimage(iplotpos,label,atitle,filename_grey,filename_contours,VisibleXaxis=False,VisibleYaxis=False,DoBeamEllipse=False,DoGreyCont=True,Clevs=[vsyst-vrange,vsyst,vsyst+vrange])
            (clevs,clabels)=addimage(iplotpos,label,atitle,filename_grey,filename_contours=filename_contours,filename_errormap=filename_errormap, filename_fiterrormap=filename_fiterrormap,VisibleXaxis=True,VisibleYaxis=False,DoBeamEllipse=True,DoGreyCont=False,vsyst=vsyst,nplotsx=nplotsx,nplotsy=nplotsy,SymmetricRange=False,MedianvalRange=False,DoCB=True, cmap=cmap_vturb,scaleim=1E-2,cbfmt='%.1f',cbunits=r'$\rm{m}\,\rm{s}^{-1}$',Zoom=Zoom,DoInterestingRegion=False,side=side)

        if RunMCMC:
            

            atitle=r'$\sigma(\Sigma_g)$'
            label=labels[ilabel]
            ilabel+=1
            filename_contours=False
            filename_grey=workdir+'errSigma_g.fits'
            filename_errormap=False
            iplotpos += 1
            #addimage(iplotpos,label,atitle,filename_grey,filename_contours,VisibleXaxis=False,VisibleYaxis=True,DoBeamEllipse=False,DoGreyCont=True,Clevs=[vsyst-vrange,vsyst,vsyst+vrange])
            (clevs,clabels)=addimage(iplotpos,label,atitle,filename_grey,filename_contours=filename_contours,filename_errormap=filename_errormap, filename_fiterrormap=filename_fiterrormap,VisibleXaxis=True,VisibleYaxis=True,DoBeamEllipse=True,DoGreyCont=False,vsyst=vsyst,nplotsx=nplotsx,nplotsy=nplotsy,SymmetricRange=False,MedianvalRange=10.,DoCB=True, cmap=cmap_Sigmag,cbfmt='%.2f',cbunits=r' '+SigmaScale[1]+r'$\rm{g}\,\rm{cm}^{-2}$',Zoom=Zoom,DoInterestingRegion=False,side=side,scaleim=SigmaScale[0])

            if (not fix_T):
                atitle=r'$\sigma(T)$'
                label=labels[ilabel]
                ilabel+=1
                filename_contours=False
                filename_grey=workdir+'errtemperature.fits'
                filename_errormap=False
                iplotpos += 1
                #addimage(iplotpos,label,atitle,filename_grey,filename_contours,VisibleXaxis=False,VisibleYaxis=False,DoBeamEllipse=False,DoGreyCont=True,Clevs=[vsyst-vrange,vsyst,vsyst+vrange])
                (clevs,clabels)=addimage(iplotpos,label,atitle,filename_grey,filename_contours=filename_contours,filename_errormap=filename_errormap, filename_fiterrormap=filename_fiterrormap,VisibleXaxis=True,VisibleYaxis=False,DoBeamEllipse=True,DoGreyCont=False,vsyst=vsyst,nplotsx=nplotsx,nplotsy=nplotsy,SymmetricRange=False,MedianvalRange=10.,DoCB=True, cmap='YlOrBr',scaleim=1.,cbfmt='%.1f',cbunits=r'$\rm{K}$',Zoom=Zoom,DoInterestingRegion=False,side=side)

            atitle=r'$\sigma(v^\circ)$'
            label=labels[ilabel]
            ilabel+=1
            filename_contours=False
            filename_grey=workdir+'errvelocentroid.fits'
            filename_errormap=False
            iplotpos += 1
            #addimage(iplotpos,label,atitle,filename_grey,filename_contours,VisibleXaxis=False,VisibleYaxis=False,DoBeamEllipse=False,DoGreyCont=True,Clevs=[vsyst-vrange,vsyst,vsyst+vrange])
            (clevs,clabels)=addimage(iplotpos,label,atitle,filename_grey,filename_contours=filename_contours,filename_errormap=filename_errormap, filename_fiterrormap=filename_fiterrormap,VisibleXaxis=True,VisibleYaxis=False,DoBeamEllipse=True,DoGreyCont=False,vsyst=vsyst,nplotsx=nplotsx,nplotsy=nplotsy,SymmetricRange=False,MedianvalRange=5.,DoCB=True, cmap='Greys',scaleim=1E3,cbfmt='%.1f',cbunits=r'$\rm{m}\,\rm{s}^{-1}$',Zoom=Zoom,DoInterestingRegion=False,side=side)


            if (not fix_vturb):
                atitle=r'$\sigma(v_\mathrm{turb})$'
                label=labels[ilabel]
                ilabel+=1
                filename_contours=False
                filename_grey=workdir+'errvturb.fits'
                filename_errormap=False
                iplotpos += 1
                #addimage(iplotpos,label,atitle,filename_grey,filename_contours,VisibleXaxis=False,VisibleYaxis=False,DoBeamEllipse=False,DoGreyCont=True,Clevs=[vsyst-vrange,vsyst,vsyst+vrange])
                (clevs,clabels)=addimage(iplotpos,label,atitle,filename_grey,filename_contours=filename_contours,filename_errormap=filename_errormap, filename_fiterrormap=filename_fiterrormap,VisibleXaxis=True,VisibleYaxis=False,DoBeamEllipse=True,DoGreyCont=False,vsyst=vsyst,nplotsx=nplotsx,nplotsy=nplotsy,SymmetricRange=False,MedianvalRange=10.,DoCB=True, cmap=cmap_vturb,scaleim=1E-2,cbfmt='%.1f',cbunits=r'$\rm{m}\,\rm{s}^{-1}$',Zoom=Zoom,DoInterestingRegion=False,side=side)


                



        plt.subplots_adjust(hspace=0.1)
        plt.subplots_adjust(wspace=0.02)


        print( fileout)
        #plt.tight_layout()

        print( "USED VSYST=",vsyst)
        plt.savefig(fileout, bbox_inches='tight', dpi=300)


        #plt.savefig(fileout)

        return



