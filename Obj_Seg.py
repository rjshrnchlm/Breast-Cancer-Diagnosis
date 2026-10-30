from Evaluation import net_evaluation
from Global_Vars import Global_Vars
import numpy as np
from Model_Regiongrowing import model_region_growing


def objfun_seg(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            predict = model_region_growing(Feat, sol=sol)
            EVAl = []
            for img in range(len(predict)):
                Eval = net_evaluation(predict[img], Tar[img])
                EVAl.append(Eval)
            mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
            Fitn[i] = 1 / (mean_EVAl[0, 5])  # IoU
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        predict = model_region_growing(Feat, sol=sol)
        EVAl = []
        for img in range(len(predict)):
            Eval = net_evaluation(predict[img], Tar[img])
            EVAl.append(Eval)
        mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
        Fitn = 1 / (mean_EVAl[0, 5])  # IoU
        return Fitn