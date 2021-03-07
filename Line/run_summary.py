import sys
include_path='/home/simon/common/python/include/'
sys.path.append(include_path)

import Slab.Line.SummaryLineFit as SummaryLineFit

workdirs=['output_iminuit_multiso/',]
#workdirs=['output_iminuit_2iso/',]
vsyst= 1.2
for aworkdir in workdirs:
        fileout = aworkdir+'fig_summary.pdf'
        fix_vturb=False
        SummaryLineFit.exec_summary(aworkdir,fileout,vsyst=vsyst, vrange=10.,fix_vturb=fix_vturb, WCont=False, Zoom=False)
