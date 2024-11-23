import os
import re
import numpy as np
import sys

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/'
sys.path.append(include_path)

import Slab.Continuum.src.SummaryFig  as SummaryFig

datadir = 'output_imoptim_noCG/'
#datadir = 'mockdata_Bump_DSHARP/'

outputdir = datadir

files_images = [
    'B6_SB17LB19_r2_degrad_to_b8_z_resamp.fits', 'B8_r1_zoom_resamp.fits'
]


titles = [
    r'B6', r'B8'
]
fileout = outputdir + 'fig_intensityimages.png'

SummaryFig.exec_summary(
    datadir,
    files_images,
    fileout,
    titles=titles,
    outputdir=outputdir,
    WithModels=True,
    ilabelstart=0,
    Zoom=False,
    side=1.5)

#files_images = [
#    'specindex_92.fits', 'specindex_126.fits', 'specindex_223.fits',
#    'specindex_336.fits'
#]
#titles = [
#    r'$\alpha(92\,$GHz)', r'$\alpha(126\,$GHz)', r'$\alpha(223\,$GHz)',
#    r'$\alpha(336\,$GHz)'
#]
#fileout = outputdir + 'fig_summary_specindex.png'
#
#SummaryFig.exec_summary(
#    datadir,
#    files_images,
#    fileout,
#    titles=titles,
#    #WithModels=outputdir,  # False
#    IsSpecIndex=True,
#    WithModels=False,
#    ilabelstart=0,
#    Zoom=True,
#    side=1.)
#
