import numpy as np
from Global_Vars import Global_Vars
from Model_ENSEMBLE import Model_ENSEMBLE


def Objfun(Soln):
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)
        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        Feat = Global_Vars.Feat
        Tar = Global_Vars.Target
        Selected_Feat, Eval, pred = Model_ENSEMBLE(Feat, Tar, sol.astype('int'))
        Fitn[i] = 1 / (Eval[4])  # Accuracy
    return Fitn