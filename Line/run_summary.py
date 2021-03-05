
import SummaryLineFit_wHband

#workdirs=['output_iminuit_multiso/',]
workdirs=['output_iminuit_2iso/',]
vsyst= 7.2
for aworkdir in workdirs:
        fileout = aworkdir+'fig_summary.pdf'
        fix_vturb=False
        SummaryLineFit_wHband.exec_summary(aworkdir,fileout,vsyst=vsyst, vrange=10.,fix_vturb=fix_vturb, WCont=False, Zoom=False)
