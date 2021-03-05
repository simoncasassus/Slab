from astropy.io import fits 
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import constants as const


#cmap='RdBu_r'
#cmap='ocean_r'
# cmap='viridis'
# cmap='RdYlBu_r'
        
#        # -----------------------------------------------------------
#        # nice fonts
#        # -----------------------------------------------------------
#        matplotlib.rc('font', family='sans-serif') 
#        matplotlib.rcParams.update({'font.size': 12})

include_path='/Users/simon/common/python/include/'
sys.path.append(include_path)
import ImUtils.Resamp as Resamp
import ImUtils.Cube2Im as Cube2Im

c_light=const.c.value


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


def addprofile(ax, hducubeobs, hducubemod, label, xoffset=0.05, yoffset=0.2, VisibleXaxis=True, VisibleYaxis=True, restfreq=230.5380000*1E9,speccolor='black',specoffset=0.,legendlabel=''):


    #fig_axspec.setp(ax.get_xticklabels(),visible=True) #, fontsize=6)
    #fig_axspec.setp(ax.get_yticklabels(),visible=True) #, fontsize=6)

    hdrobs=hducubeobs[0].header
    cubeobs=1E3*hducubeobs[0].data
    hdrmod=hducubemod[0].header
    cubemod=1E3*hducubemod[0].data

    nus=((np.arange(hdrobs['NAXIS3'])-hdrobs['CRPIX3']+1)*hdrobs['CDELT3'])+hdrobs['CRVAL3']
    nusmod=((np.arange(hdrmod['NAXIS3'])-hdrmod['CRPIX3']+1)*hdrmod['CDELT3'])+hdrmod['CRVAL3']

    print("c_light",c_light)

    vs=-(nus-restfreq)*c_light*1E-3/restfreq
    vsmod=-(nusmod-restfreq)*c_light*1E-3/restfreq
         
    print("nus",nus)
    print("vs",vs)
          
    ioff=int(((xoffset/3600.)/hdrobs['CDELT1'])+(hdrobs['CRPIX1']-1))
    joff=int(((yoffset/3600.)/hdrobs['CDELT2'])+(hdrobs['CRPIX2']-1))

    print("xoffset ",xoffset," yoffset", yoffset, " -->")
    print("ioff    ",ioff,"   joff", joff)
    
    ioffmod=int(((xoffset/3600.)/hdrmod['CDELT1'])+(hdrmod['CRPIX1']-1))
    joffmod=int(((yoffset/3600.)/hdrmod['CDELT2'])+(hdrmod['CRPIX2']-1))

    if (ioffmod !=  ioff):
        print("mismatched canvases")
    if (joffmod !=  joff):
        print("mismatched canvases")
    
    specobs=cubeobs[:,joff,ioff]
    specmod=cubemod[:,joffmod,ioffmod]
    
    
    ax.plot(vs,specobs+specoffset,color=speccolor,linewidth=0.7,linestyle='solid',label=legendlabel)
    ax.plot(vsmod,specmod+specoffset,color=speccolor,linewidth=2.0,linestyle='dashed',alpha=0.5)
    
    #ymin=np.min(specobs)
    #ymax=np.max(specobs)
    #ax.text(np.min(vs),ymin+0.9*(ymax-ymin),label,weight='bold',fontsize=12)
    
            

    if (VisibleYaxis):
        ax.set_ylabel(r'mJy beam$^{-1}$')
        
    if (VisibleXaxis):
        ax.set_xlabel(r'km s$^{-1}$')
        
    
    
    ax.legend()


def addimage(ax,label,atitle,filename_grey,filename_contours,filename_errormap=False, filename_fiterrormap=False,VisibleXaxis=False,VisibleYaxis=True,DoBeamEllipse=False,DoGreyCont=False,vsyst=0.,SymmetricRange=False,MedianvalRange=False,DoCB=True,cmap='RdBu_r',MedRms=True,Zoom=False,scaleim=1.,cbfmt="%.1e",cbunits='',Side=1.5,markers=[]):

    plt.setp(ax.get_xticklabels(),visible=VisibleXaxis)
    plt.setp(ax.get_yticklabels(),visible=VisibleYaxis)

    ax.tick_params(axis='both',length = 5, width=1., color = 'grey',direction='in',left=True, right=True,bottom=True, top=True)

    ax.spines['right'].set_color('grey')
    ax.spines['left'].set_color('grey')
    ax.spines['top'].set_color('grey')
    ax.spines['bottom'].set_color('grey')




    ax.set_ylabel(r'$\delta$  offset / arcsec')
    ax.set_xlabel(r'$\alpha$ offset / arcsec')
            

    print( "loading filename_grey",filename_grey)

    fin = fits.open(filename_grey)
    fin = Cube2Im.slice0(fin)
    im_grey = fin.data
    hdr_grey= fin.header
    cdelt=3600.*hdr_grey['CDELT2']

    side0=hdr_grey['NAXIS2']*cdelt


    if Zoom:
            side=Side 
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


                    hduzoom=Resamp.gridding(fin,hdrzoom,ReturnHDU=True)
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

    print( "loading filename_grey",filename_fiterrormap)

    f = fits.open(filename_fiterrormap)
    f = Cube2Im.slice0(f)
    im_fiterrormap = f.data
    hdr_fiterrormap= f.header
    if Zoom:
        Resample=True
        if Resample:
            hduzoom_fiterrmap=Resamp.gridding(f,hdrzoom,ReturnHDU=True)
            subim_fiterrormap=hduzoom_fiterrmap.data
        else:
            subim_fiterrormap=im_fiterrormap[j0:j1,i0:i1]
    else:
            subim_fiterrormap=im_fiterrormap.copy()

    # medsubim_fiterrormap=medfilt2d(subim_fiterrormap,kernel_size=11)

    #Vtools.View(subim_fiterrormap)
    #Vtools.View(medsubim_fiterrormap)

    typicalerror=np.median(subim_fiterrormap)

    #mask=np.where((subim_fiterrormap-medsubim_fiterrormap) > 3.* medsubim_fiterrormap)
    mask=np.where(subim_fiterrormap<20.*typicalerror)

    immask=np.zeros(subim_fiterrormap.shape)
    immask[mask]=1
    print("number of pixels masked:",np.sum(immask))
    #print("viewing fiterrormap immask")
    #Vtools.View(immask)


    #plt.figure(2)
    #plt.imshow(immask)
    #plt.show()
    #plt.figure(1)

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



    if SymmetricRange:
            range2=vsyst+SymmetricRange
            range1=vsyst-SymmetricRange
            # subim_grey[np.where(subim_grey < range1)]=vsyst
            # subim_grey[np.where(subim_grey > range2)]=vsyst
            clevs = [range1,0.,range2]
            clabels=['%.0f' % (clevs[0]),'','%.0f' % (clevs[2])]
    elif MedianvalRange:
            typicalvalue=np.median(subim_grey[mask])
            rms=np.std(subim_grey[mask])
            medrms=np.sqrt(np.median( (subim_grey[mask] - typicalvalue)**2))

            print("typical value ",typicalvalue," rms ",rms,"medrms",medrms)
            range1=np.min(subim_grey[mask])
            if MedRms:
                    imagerms=medrms
            else:
                    imagerms=rms
            range2=typicalvalue+10.*imagerms
            clevs = [range1,range2]
            clabels=['%.1f' % (clevs[0]),'%.1f' % (clevs[1])]
    else:
            range2=np.max(subim_grey[mask])
            range1=np.min(subim_grey[mask])
            clevs = [range1,range2]
            clabels=['%.1f' % (clevs[0]),'%.1f' % (clevs[1])]


    if ('sigma' in filename_grey):
            cmap='magma_r'

    print("max:",np.max(subim_grey))
    print("min:",np.min(subim_grey))
    print("range1",range1,"range2",range2)
    if (np.isnan(subim_grey).any()):
            print("NaNs in subim_grey")
    subim_grey=np.nan_to_num(subim_grey)


    theimage=ax.imshow(subim_grey, origin='lower', cmap=cmap, #norm=norm,
               extent=[a0,a1,d0,d1], vmin=range1, vmax=range2, interpolation='nearest') #'nearest'  'bicubic'

    #plt.plot(0.,0.,marker='*',color='yellow',markersize=0.2,markeredgecolor='black')
    ax.plot(0.,0.,marker='*',color='yellow',markersize=4.)


    print("label:",label)
    if (len(markers) > 0):
        for ipos, aposition in enumerate(markers):
            xoffset=aposition[1]
            yoffset=aposition[0]
            print("adding a cross at ",xoffset,yoffset, "ipos ",ipos)
            ax.plot(xoffset,yoffset,marker='x',color='white',markersize=10)
            #ax.text(xoffset,yoffset,str(ipos),weight='bold',fontsize=12,bbox=dict(facecolor='white', alpha=0.8))
            ax.text(xoffset,yoffset,str(ipos),weight='bold',fontsize=12, color='white',alpha=0.8)
        

    
    ax.text(a1*0.9,d0*0.9,atitle,weight='bold',fontsize=12,ha='right',bbox=dict(facecolor='white', alpha=0.8))

    #ax.text(a0*0.9,d0*0.9,label,weight='bold',fontsize=12,bbox=dict(facecolor='white', alpha=0.8))

    axcb=plt.gca()

    if (DoCB):
            cb=colorbar(theimage,cbfmt=cbfmt)
            cb.ax.tick_params(labelsize='small')
            print("CB label",cbunits)
            cb.set_label(cbunits)



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






    return  clevs, clabels




        



matplotlib.rc('font', family='sans-serif') 
matplotlib.rcParams.update({'font.size': 10})
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

size_marker=10
        
#cmaps = ['magma', 'inferno', 'plasma', 'viridis', 'bone', 'afmhot', 'gist_heat', 'CMRmap', 'gnuplot', 'Blues_r', 'Purples_r', 'ocean', 'hot', 'seismic_r']
gamma=1.0

#fig = plt.figure(constrained_layout=True,figsize=(10, 15))
fig = plt.figure(constrained_layout=True,figsize=(9, 14))
#gs = fig.add_gridspec(2, 3, width_ratios=[2., 1., 1.], height_ratios=[1.,1.])
#gs = fig.add_gridspec(7, 2, width_ratios=[1., 1.], height_ratios=[1.5,.6,.6,.6,.6,.6,.6])
#gs = fig.add_gridspec(4, 2, width_ratios=[1., 1.], height_ratios=[1.2,.6,.6,.6])
gs = fig.add_gridspec(2, 2, width_ratios=[1., 1.], height_ratios=[.8,1.2])


fig_ax1 = fig.add_subplot(gs[0,:])

#alloffsets=[ [0.2,0.05], [0.1,0.12], [-0.1,-0.15], [-0.2,0.]]  # extraction point in arcsecs, y, x 


alloffsets=[ [-0.1,-0.15], [-0.2,0.]]  # extraction point in arcsecs, y, x 

workdir='/strelka_ssd/simon/HD135344B/Dmoments/12CO_sgaussminuit_nobase_z/'
obscube='/strelka_ssd/simon/HD135344B/red/tclean_contsubHD135344Bbriggs2.0_12CO_z.fits'
modelcube='/strelka_ssd/simon/HD135344B/linefit/output_iminuit_multiso/model_CO.fits'
restfreq=230.5380000*1E9


filename_fiterrormap=workdir+'fiterrormap.fits'
cmap='ocean_r'
atitle=r'$I_\mathrm{gauss}$'
label='a'
filename_contours=False
filename_grey=workdir+'im_gmom_0.fits'
filename_errormap=False
vsyst=7.2
(clevs,clabels)=addimage(fig_ax1,label,atitle,filename_grey,filename_contours=filename_contours,filename_errormap=filename_errormap, filename_fiterrormap=filename_fiterrormap,VisibleXaxis=True,VisibleYaxis=True,DoBeamEllipse=True,DoGreyCont=False,vsyst=vsyst,SymmetricRange=False,DoCB=True, cmap=cmap,scaleim=1E3,cbfmt='%.1f',cbunits=r'$\rm{mJy}\,\rm{beam}^{-1} \,\rm{km}\,\rm{s}^{-1}$',Zoom=False,markers=alloffsets)


#allpositions=[ [0,1], [0,2], [1,1], [1,2]]   # gs positions for profiles
#allpositions=[ [1,0], [2,0], [1,1], [2,1]]   # gs positions for profiles
allpositions=[ [1,0], [1,1]]   # gs positions for profiles

for ioffset  in list(range(len(alloffsets))):
    yoffset=alloffsets[ioffset][0]
    xoffset=alloffsets[ioffset][1]
    plotpos=allpositions[ioffset]

    print("plotpos",plotpos)
    
    fig_axspec = fig.add_subplot(gs[plotpos[0],plotpos[1]])

    
    fig_axspec.set_title(str(ioffset)+': '+r'$\alpha='+str(xoffset)+' \,\,\,\delta='+str(yoffset)+'$')
    
    label=ioffset
    
    hducubemod=fits.open(modelcube)
    hducubeobs=fits.open(obscube)


    VisibleXaxis=False
    VisibleYaxis=False
        
    addprofile(fig_axspec,hducubeobs,hducubemod,label,xoffset=xoffset,yoffset=yoffset,VisibleXaxis=VisibleXaxis,VisibleYaxis=VisibleYaxis,restfreq=restfreq,specoffset=0.,speccolor='C0',legendlabel=r'$^{12}$CO(2-1)')



workdir='/strelka_ssd/simon/HD135344B/Dmoments/13CO_sgaussminuit_nobase_z/'
obscube='/strelka_ssd/simon/HD135344B/red/tclean_contsubHD135344Bbriggs2.0_13CO_z.fits'
modelcube='/strelka_ssd/simon/HD135344B/linefit/output_iminuit_multiso/model_13C16O.fits'
restfreq=220.3986841281*1E9

#allpositions=[ [1,0], [2,0], [1,1], [2,1]]   # gs positions for profiles
#allpositions=[ [3,0], [4,0], [3,1], [4,1]]   # gs positions for profiles
#allpositions=[ [2,0], [4,0], [3,1], [4,1]]   # gs positions for profiles
#allpositions=[ [2,0], [2,1]]   # gs positions for profiles

for ioffset  in list(range(len(alloffsets))):
    yoffset=alloffsets[ioffset][0]
    xoffset=alloffsets[ioffset][1]
    plotpos=allpositions[ioffset]

    print("plotpos",plotpos)
    
    fig_axspec = fig.add_subplot(gs[plotpos[0],plotpos[1]])

    
    #fig_axspec.set_title(r'$\alpha='+str(xoffset)+' \delta='+str(yoffset)+'$')
    
    label=ioffset
    
    hducubemod=fits.open(modelcube)
    hducubeobs=fits.open(obscube)


    VisibleXaxis=False
    VisibleYaxis=False
        
    addprofile(fig_axspec,hducubeobs,hducubemod,label,xoffset=xoffset,yoffset=yoffset,VisibleXaxis=VisibleXaxis,VisibleYaxis=VisibleYaxis,restfreq=restfreq,specoffset=-10.,speccolor='C1',legendlabel=r'$^{13}$CO(2-1)')

workdir='/strelka_ssd/simon/HD135344B/Dmoments/C18O_sgaussminuit_nobase_z/'
obscube='/strelka_ssd/simon/HD135344B/red/tclean_contsubHD135344Bbriggs2.0_C18O_z.fits'
modelcube='/strelka_ssd/simon/HD135344B/linefit/output_iminuit_multiso/model_C18O.fits'
restfreq=219.5603541*1E9




#f3_ax1.set_title('gs[0, :]')
#f3_ax2 = fig3.add_subplot(gs[1, :-1])
#f3_ax2.set_title('gs[1, :-1]')
#f3_ax3 = fig3.add_subplot(gs[1:, -1])
#f3_ax3.set_title('gs[1:, -1]')
#f3_ax4 = fig3.add_subplot(gs[-1, 0])
#f3_ax4.set_title('gs[-1, 0]')
#f3_ax5 = fig3.add_subplot(gs[-1, -2])
#f3_ax5.set_title('gs[-1, -2]')
#


#allpositions=[ [5,0], [6,0], [5,1], [6,1]]   # gs positions for profiles
#allpositions=[ [3,0], [3,1]]   # gs positions for profiles


for ioffset  in list(range(len(alloffsets))):
    yoffset=alloffsets[ioffset][0]
    xoffset=alloffsets[ioffset][1]
    plotpos=allpositions[ioffset]

    print("plotpos",plotpos)
    
    fig_axspec = fig.add_subplot(gs[plotpos[0],plotpos[1]])

    
    #fig_axspec.set_title(r'$\alpha='+str(xoffset)+' \delta='+str(yoffset)+'$')
    
    label=ioffset
    
    hducubemod=fits.open(modelcube)
    hducubeobs=fits.open(obscube)


    VisibleXaxis=True
    VisibleYaxis=False
    if (plotpos[1] ==  0):
        VisibleYaxis=True
        
    addprofile(fig_axspec,hducubeobs,hducubemod,label,xoffset=xoffset,yoffset=yoffset,VisibleXaxis=VisibleXaxis,VisibleYaxis=VisibleYaxis,restfreq=restfreq,specoffset=-20.,speccolor='C2',legendlabel=r'C$^{18}$O(2-1)')

plt.legend()
plt.subplots_adjust(hspace=0.15)
plt.subplots_adjust(wspace=0.2)
        
fileout = 'fig_comparespectra_alliso.pdf'
print( fileout)
#plt.tight_layout()

plt.savefig(fileout, bbox_inches='tight')
#plt.savefig(fileout)


