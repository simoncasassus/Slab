import Linefit_iminuit
import MolData

maxradius = 0.5 #0.4  # arcsec, radius inside  of which to make the fit
J_up=2
ncores = 30                
    
# order files from most optically thick to most optically thin
inputcubefiles = [
    '/strelka_ssd/simon/HD135344B/red/tclean_contsubHD135344Bbriggs2.0_12CO.fits',  
    '/strelka_ssd/simon/HD135344B/red/tclean_contsubHD135344Bbriggs2.0_13CO.fits',
    '/strelka_ssd/simon/HD135344B/red/tclean_contsubHD135344Bbriggs2.0_C18O.fits']

moldatafiles=['./LAMDAmoldatafiles/molecule_12c16o.inp',
              './LAMDAmoldatafiles/molecule_13c16o.inp',
              './LAMDAmoldatafiles/molecule_12c18o.inp']

outputdir='./output_iminuit_multiso_dev/'

Linefit_iminuit.exec_optim(inputcubefiles,maxradius=maxradius,moldatafiles=moldatafiles,J_up=J_up,ncores=ncores,outputdir=outputdir,ViewIndividualSpectra=False,Fix_vturbulence=False)







