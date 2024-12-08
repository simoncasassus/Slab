import sys
import numpy as np
import os
import os.path
from time import time
from copy import deepcopy
import astropy.constants as const

from .Likelihood import logL


class Merit():
    """
    call Likelihood
    """
    def __init__(
        self,
        with_specindexdata=False,
        VerboseInit=False,
        ExecTimeReport=False,
        LbdaSigma_gRegul=1.,
        MaxOptDepth=3.,
        LbdaOptDepthRegul=1E3,
        LbdaamaxRegul=1.,
        Regul=False,
    ):
        """
        if Regul then use weights Lbda*, MaxOptDepth is at lowest freq
        """
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            if VerboseInit:
                print("Merit> setting ", a_attribute, " to ",
                      initlocals[a_attribute])
            setattr(self, a_attribute, initlocals[a_attribute])

    # self.__dict__.update(ASED.__dict__)

    def calcul(self, ZSetup, ZData, ASED, ASED4alphas=None):
        if self.ExecTimeReport:
            time_start = time()

        ASED.calcul()
        if self.ExecTimeReport:
            time_end_1 = time()
            print("time for ASED calcul: ", time_end_1 - time_start, " s")

        if self.with_specindexdata:
            ASED4alphas.calcul()
            if self.ExecTimeReport:
                time_end_2 = time()
                print("time for ASED4alphas calcul: ", time_end_2 - time_end_1,
                      " s")

        #print("calling LogL ",ASED.Tdust,ASED.amax, ASED.Sigma_g, ASED.q_dustexpo, ASED.nus, ASED.Inus)
        retvals = logL(ZData,
                       ASED,
                       with_specindexdata=self.with_specindexdata,
                       ASED4alphas=ASED4alphas,
                       LbdaSigma_gRegul=self.LbdaSigma_gRegul,
                       MaxOptDepth=self.MaxOptDepth,
                       LbdaOptDepthRegul=self.LbdaOptDepthRegul,
                       LbdaamaxRegul=self.LbdaamaxRegul,
                       Regul=self.Regul)
        #chi2 = retvals[0]
        #print("result ",retvals[0], ASED.nus,ASED.Inus, ASED4alphas.Inus, ASED.amax, )

        if self.ExecTimeReport:
            time_end = time()
            print("total time for Likelihood evaluation: ",
                  time_end - time_start, " s")

        return retvals
