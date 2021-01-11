import Linefit_iminuit
import MolData

maxradius = 0.8 #0.4  # arcsec, radius inside  of which to make the fit
J_up=2
ncores = 30                
    
# order files from most optically thick to most optically thin
inputcubefiles = [
    '/strelka_ssd/simon/HD135344B/red/tclean_contsubHD135344Bbriggs2.0_12CO_z.fits',  
    '/strelka_ssd/simon/HD135344B/red/tclean_contsubHD135344Bbriggs2.0_13CO_z.fits',
    '/strelka_ssd/simon/HD135344B/red/tclean_contsubHD135344Bbriggs2.0_C18O_z.fits']

moldatafiles=['./LAMDAmoldatafiles/molecule_12c16o.inp',
              './LAMDAmoldatafiles/molecule_13c16o.inp',
              './LAMDAmoldatafiles/molecule_12c18o.inp']

outputdir='./output_iminuit_multiso/'

# Examine fits in individual positions:
xoffset=-0.15
yoffset=-0.1
ViewIndividualSpectra=[[yoffset,xoffset]]

# Run the whole image within maxradius:
ViewIndividualSpectra=False

Linefit_iminuit.exec_optim(inputcubefiles,maxradius=maxradius,moldatafiles=moldatafiles,J_up=J_up,ncores=ncores,outputdir=outputdir,ViewIndividualSpectra=ViewIndividualSpectra,Fix_vturbulence=False)







