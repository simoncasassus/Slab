import sys

include_path='/home/simon/common/python/include/'
sys.path.append(include_path)

#import SummaryLineFit_full
#import Slab.Line.SummaryLineFit_full as SummaryLineFit
import Slab.Line.SummaryLineFit_compact as SummaryLineFit

#workdirs=['output_iminuit_multiso/',]
workdirs=['output_iminuit_2iso/',]
workdirs=['output_iminuit_multiso_uvtaper/']
workdirs=['output_iminuit_multiso_uvtaper_emcee/','output_iminuit_multiso_uvtaper_debug/','output_iminuit_multiso/']
workdirs=['output_iminuit_multiso_uvtaper_emcee_werrors/']
#workdirs=['output_iminuit_multiso/']
vsyst= 7.2
for aworkdir in workdirs:
        #fileout = aworkdir+'fig_summary.pdf'
        fileout = aworkdir+'fig_summary_compact.pdf'
        fix_vturb=False
        SummaryLineFit.exec_summary(aworkdir,fileout,vsyst=vsyst, vrange=10.,fix_vturb=fix_vturb, WCont=False, Zoom=False)
