import numpy as np
from Evaluation import evaluation
from Model_Inception import Model_Inception
from Model_Resnet import Model_Resnet
from Model_VGG16 import Model_VGG16


def Model_ENSEMBLE(Feat, Target, sol=None):
    Selected_Feat = Feat[:, sol[:10].astype(int)]

    learnper = round(Selected_Feat.shape[0] * 0.75)
    Train_Data = Selected_Feat[learnper:, :]
    Train_Tar = Target[learnper:, :]
    Test_Data = Selected_Feat[:learnper, :]
    Test_Tar = Target[:learnper, :]

    Eval_VGG16, Pred_VGG16 = Model_VGG16(Train_Data, Train_Tar, Test_Data, Test_Tar, sol=sol[10])
    Eval_Resnet, Pred_Resnet = Model_Resnet(Train_Data, Train_Tar, Test_Data, Test_Tar, sol=sol[11])
    Eval_Inception, Pred_Inception = Model_Inception(Train_Data, Train_Tar, Test_Data, Test_Tar, sol=sol[12])

    pred = (Pred_VGG16 + Pred_Resnet + Pred_Inception) / 3
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = np.asarray(pred).astype('int')
    Eval = evaluation(Test_Tar, pred)
    return Selected_Feat, Eval, pred
