import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from astropy.coordinates import SkyCoord


file_in = 'output_imoptim_wb3r07_wfillfactor/B6_SB17LB19_r2_degrad_to_b3r07_z_resamp.fits'
hdu = fits.open(file_in)
im=hdu[0].data
hdr=hdu[0].header

indices = np.where(im == im.max())
j0,i0=indices
print("i0",i0,"j0",j0)
## w1 = WCS(hdr)
## (x0, y0) = w1.wcs_pix2world(i0,j0, 0)
## print("x0 ",x0,"y0 ",y0)
dx0 = (i0-hdr['CRPIX1']+1)*hdr['CDELT1']*3600.
dy0 = (j0-hdr['CRPIX2']+1)*hdr['CDELT2']*3600.
print("dx0",dx0,"dy0",dy0)

#A ( 18.8, 8.6) = 7.00e-03 
