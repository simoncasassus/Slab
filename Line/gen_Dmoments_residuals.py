import os
import sys
import re

import astropy.io.fits as pf
import numpy as np
from copy import copy, deepcopy


include_path=os.environ['HOME']+'/common/python/include/'
sys.path.append(include_path)
#import  DGaussMinuit 
import  GMoments.DGaussMoments  as DGaussMoments

#import SummaryLineFit
import SummaryDMoments
import time
#from time import gmtime, strftime


inputcubefiles = [
    '/strelka_ssd/simon/HD135344B/red/tclean_contsubHD135344Bbriggs2.0_12CO_z.fits',  
    '/strelka_ssd/simon/HD135344B/red/tclean_contsubHD135344Bbriggs2.0_13CO_z.fits',
    '/strelka_ssd/simon/HD135344B/red/tclean_contsubHD135344Bbriggs2.0_C18O_z.fits']

workdir='output_iminuit_multiso/'
#modelfiles = [
#    workdir+'model_CO_z.fits',  
#    workdir+'model_13C16O_z.fits',
#    workdir+'model_C18O_z.fits']
#
modelfiles = [
    workdir+'model_CO.fits',  
    workdir+'model_13C16O.fits',
    workdir+'model_C18O.fits']


for iiso in list(range(len(inputcubefiles))):
    fileresids=re.sub('model_','residual_',modelfiles[iiso])
    hduobs=pf.open(inputcubefiles[iiso])
    hdumod=pf.open(modelfiles[iiso])
    datacube=hduobs[0].data
    modelcube=hdumod[0].data
    residcube=datacube-modelcube
    hduresid=deepcopy(hduobs)
    hduresid[0].data=residcube
    print("PUNCHING ",fileresids)
    hduresid.writeto(fileresids,overwrite=True)
    

    start_time=time.time()
    print("start Curve_fit:",time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    isoname=re.search(r"model_(\w+)_?z?.fits",modelfiles[iiso])
    print("isoname ",isoname.group(1))
    workdir_resids=workdir+'moments_resids_'+isoname.group(1)+'/'
    print("now taking moments and storing in workdir ",workdir_resids)
    
    os.system("rm -rf "+workdir_resids)
    os.system('mkdir '+workdir_resids)
    DGaussMoments.exec_Gfit(fileresids,workdir_resids,wBaseline=False,n_cores=30,zoom_area=0.6,Noise=2.5E-3,Clip=True,DoubleGauss=False,StoreModel=False,Randomize2ndGauss=False,ShrinkCanvas=False,UseCommonSigma=False)

    end_time=time.time()
    print( "curve_fit done in (elapsed time):",   end_time - start_time)



    vsyst= 7.2
    fileout = workdir_resids+'fig_summary.pdf'
    


    SummaryDMoments.exec_summary(workdir_resids,fileout,vsyst=vsyst, vrange=10.,ngauss=1, WCont=False, Zoom=False,Side=0.6*2)
