import os
import re
import numpy as np
import sys

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/'
sys.path.append(include_path)

import Slab.Continuum.src.SummaryFigDust as SummaryFigDust

outputdir = './output_imoptim_b8r0_amax0.01_fine/'

domain = [
    ['log(Tdust)', np.log10(30.), [0., 3]],
    #['q_dustexpo', -3.5, [-3.99, -2.]],
    #['f_grain', 1., [0., 1.]],
    #['log(amax)', np.log10(1.), [np.log10(1E-3), np.log10(10.)]],  #cm
    ['log(Sigma_g)',
     np.log10(30.), [np.log10(1E-5), np.log10(1E3)]]
]  # g/cm2

#domain = [
#    ['log(Tdust)', np.log10(30.), [0., 3]],
#    #['q_dustexpo', -3.5, [-3.99, -2.]],
#    #['f_grain', 1., [0., 1.]],
#    #['log(amax)', np.log10(1.), [np.log10(1E-3), np.log10(10.)]],  #cm
#    ['log(Sigma_g)',
#     np.log10(30.), [np.log10(1E-5), np.log10(1E3)]]
#]  # g/cm2

errthreshs = [['log(Tdust)', 0.1], ['log(Sigma_g)', 1.]]

#fileout = outputdir + 'fig_dustparams.png'
#SummaryFigDust.exec_summary(outputdir,
#                            domain,
#                            fileout,
#                            errthresh4mask=0.1,
#                            DoInterestingRegion=True,
#                            Zoom=False,
#                            side=1.5)

fileout = outputdir + 'fig_dustparams.png'
SummaryFigDust.exec_summary(outputdir,
                            domain,
                            fileout,
                            DoInterestingRegion=True,
                            errthreshs=errthreshs)

fileout = outputdir + 'fig_dustparams_linear.png'
SummaryFigDust.exec_summary(outputdir,
                            domain,
                            fileout,
                            DoInterestingRegion=True,
                            LinearNotLog=True,
                            errthreshs=errthreshs)
