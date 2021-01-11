import numpy as np
import sys
import astropy.constants as const

#--------------------------------------------------------------------------------
#constants in cgs units
#--------------------------------------------------------------------------------

m_e = const.m_e.cgs.value       # electron mass
q_e = const.e.value				# electron charge

h_P = const.h.cgs.value
c_light = const.c.cgs.value
k_B = const.k_B.cgs.value
mp = const.m_p.cgs.value
N_A=const.N_A.value


def load_moldata(inputfile):
    """
    Return arrays with energy and weights of molecule from inputfile.
    inputfile: LAMDA file 
    """
    arch = open(inputfile,'r')
    
    arch.readline()
    molname=arch.readline().split()[0]
    arch.readline()
    molweight=float(arch.readline().split()[0])
    arch.readline()
    nlevels=int(arch.readline().split()[0])
    arch.readline()

    MasterMolData={}
    MasterMolData['name']=molname
    MasterMolData['molweight']=molweight  # g/mol 
    MasterMolData['molecularmass']=molweight/N_A # g

    levels = [] #np.arange(nlevels)
    levelenergies = [] #np.zeros(nlevels)
    levelJs = [] #np.zeros(nlevels)
    g_Js = [] #np.zeros(nlevels)
    
    for i in range(nlevels):
        l = arch.readline().split()
        levels.append(int(l[0]))
        levelenergies.append(float(l[1])* c_light * h_P)  #cm^-1 to eV
        g_Js.append(float(l[2]))
        levelJs.append(float(l[3]))


    # max temperature:
    #Emax=levelenergies[-1]
    #g_J_max=g_Js[-1]
    #epsilon=1E-2 # some small number for epsilon = g_J * exp(-E/kT)
    #Tmax = -(Emax/k_B)* np.log(epsilon / g_J_max)
    #print("OK v=0 up to T= ",Tmax," for Emax = ",Emax," g_J_max ", g_J_max)
        

    MasterMolData['levelenergies']=levelenergies
    MasterMolData['g_Js']=g_Js
    MasterMolData['levelJs']=levelJs
    MasterMolData['levelnumbers']=levels
    
    arch.readline()
    ntransitions=int(arch.readline().split()[0])
    arch.readline()

    MasterMolData['transitions']={} 
    
    for i in range(ntransitions):
        l = arch.readline().split()
        transitionnumber = l[0]
        linedata={}
        linedata['nlevelup']=int(l[1])
        linedata['nleveldown']=int(l[2])
        linedata['Einstein_A']=float(l[3])
        linedata['restfreq']=float(l[4])*1E9
        A21=linedata['Einstein_A']
        restfreq=linedata['restfreq']
        B21=A21 / ((2. * h_P * restfreq**3)/ c_light**2)
        linedata['Einstein_B21']=B21
        MasterMolData['transitions'][transitionnumber]=linedata
            
    arch.close()
    
    
    return MasterMolData


        


#-----------------------------------------------------------------------------
#
#line absorption cross section derived from Einstein Coefficients in [cm^2/s]
#
#-----------------------------------------------------------------------------
def B_21(A_21,nu):
        """
        input Einstein A and restfreq in Hz
        Return B value for stimulated radiative de-excitations
        """

        B_21 = A_21 / ((2. * h_P * nu**3)/ c**2)
        
        return B_21


#-----------------------------------------------------------------------------
#
#molecular fraction of the isotopologues
#
#-----------------------------------------------------------------------------
def molecular_fraction(name):
        if name=="C18O":
                g=(1/500.0)
        elif name=="CO":
                g=1.0
        elif name=="13C16O":
                g=1/70.0
        else:
                g=1.
        return g
