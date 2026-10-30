import numpy as np
from random import uniform
from HBA import HBA
from KHO import KHO
from Model_BILSTM import Model_BILSTM
from Model_CNN import Model_CNN
from Model_DCN import Model_DCN
from Model_Inception import Model_Inception_Feat
from Model_OELRC import Model_OELRC
from Model_Regiongrowing import model_region_growing
from Model_Resnet import Model_Resnet_Feat
from Model_VGG16 import Model_VGG16_Feat
from Obj_Seg import objfun_seg
from Objective_Function import Objfun
from Global_Vars import Global_Vars
from numpy import matlib
import cv2 as cv
import os
import pandas as pd
from POA import POA
from PROPOSED import PROPOSED
from RG import regionGrow
from Image_Results import Image_Results
from Plot_Roc import Plot_ROC
from Plot_Results import plot_Fitness, plot_results_kfold
from ZOA import ZOA
from objfun import objfun_cls


class Point(object):
 def __init__(self,x,y):
  self.x = x
  self.y = y


no_of_dataset = 3


# Read Dataset and Pre-Processing
an = 0
if an == 1:
    original = []
    Target=[]
    preproc = []
    Directory = './Dataset/'
    out_folder = os.listdir(Directory)
    for i in range(len(out_folder)):
        Orig=[]
        Tar=[]
        if i == 0:
            in_folder = Directory + out_folder[i] + '/'
            in_dir = os.listdir(in_folder)
            for j in range(len(in_dir)):
                if '.txt' in in_dir[j]:
                    pass
                elif '.pgm' in in_dir[j]:
                    filename = in_folder + in_dir[j]
                    Data = cv.imread(filename)
                    Data1 = cv.resize(Data, (128, 128),interpolation=cv.INTER_NEAREST)
                    Tar.append([])
                    Orig.append(Data1)
        else:
            sub_folder = os.listdir(Directory + out_folder[i])
            sub_folders = os.listdir(Directory + out_folder[i] + '/' + sub_folder[0])
            for j in range(len(sub_folders)): # For all Sub Folders
                sub_Files = os.listdir(Directory + out_folder[i] + '/' + sub_folder[0] + '/' + sub_folders[j])
                for k in range(len(sub_Files)):
                    print(i, j, len(sub_Files))
                    filename = Directory + out_folder[i] + '/' + sub_folder[0] + '/' + sub_folders[j] + '/' + sub_Files[k]
                    Data = cv.imread(filename)
                    Data1 = cv.resize(Data, (128, 128), interpolation=cv.INTER_NEAREST)
                    Orig.append(Data1)
                    Tar.append(j)
        original.append(Orig)
        Target.append(Tar)
    np.save('original.npy', original)
    np.save('Label.npy', Target)


# Read Target
an = 0
if an == 1:
    Target =[]
    orig = np.load('original.npy', allow_pickle=True)
    Lab = np.load('Label.npy', allow_pickle=True)
    for n in range(len(orig)):
        if n == 0:
            Excel = pd.read_excel('data.xlsx')
            Tar = np.asarray(Excel['G CLA'])
            ind = np.where(Tar == 'NORM')
            Tar = np.ones((len(Tar))).astype('int')
            Tar[ind] = 0
            Tar = Tar[:len(orig[n])]
            Tar = Tar.reshape(-1, 1).astype('int')
        else:
            Tar = np.asarray(Lab[n]).reshape(-1, 1).astype('int')
        Target.append(np.asarray(Tar))
    np.save('Target.npy', Target)

# GT
an = 0
if an == 1:
    GT = []
    original = np.load('original.npy', allow_pickle=True)
    for i in range(len(original)): # For all Datasets
        Seg=[]
        for j in range(len(original[i])):  # For all Images
            print(i, j, len(original[i]))
            gray = cv.cvtColor(original[i][j], cv.COLOR_BGR2GRAY)
            # Region Growing
            seeds = [Point(10, 10), Point(82, 120), Point(20, 100)]  # [Point(0, 255)]
            binaryImg = regionGrow(gray, seeds, 5)  # K-Means Region Growing
            region = np.zeros((binaryImg.shape)).astype(np.uint8)
            pn = np.where(binaryImg == 0)
            region[pn] = 255
            Seg.append(region)
        GT.append(Seg)
    np.save('GT.npy', GT)

# Optimization for Segmentation
an = 0
if an == 1:
    FITNESS = []
    BESTSOL = []
    for n in range(no_of_dataset):
        Feat = np.load('original.npy', allow_pickle=True)[n]  # Load the Data
        Target = np.load('Target.npy', allow_pickle=True)[n]  # Load the Target
        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 2  # Seed, Threshold value
        xmin = matlib.repmat(np.asarray([50, 5]), Npop, 1)
        xmax = matlib.repmat(np.asarray([240, 40]), Npop, 1)
        fname = objfun_seg
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("POA...")
        [bestfit1, fitness1, bestsol1, time1] = POA(initsol, fname, xmin, xmax, Max_iter)  # POA

        print("HBA...")
        [bestfit2, fitness2, bestsol2, time2] = HBA(initsol, fname, xmin, xmax, Max_iter)  # HBA

        print("ZOA...")
        [bestfit3, fitness3, bestsol3, time3] = ZOA(initsol, fname, xmin, xmax, Max_iter)  # ZOA

        print("KHO...")
        [bestfit4, fitness4, bestsol4, time4] = KHO(initsol, fname, xmin, xmax, Max_iter)  # KHO

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

        BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

        FITNESS.append(fitness)
        BESTSOL.append(BestSol_CLS)
    np.save('Fitness_Seg.npy', np.asarray(FITNESS))
    np.save('BestSol_seg.npy', np.asarray(BESTSOL))  # Save the Bestsol

# Segmentation
an = 0
if an == 1:
    Segmented = []
    for n in range(no_of_dataset):
        Feat = np.load('original.npy', allow_pickle=True)[n]  # Load the Data
        Target = np.load('GT.npy', allow_pickle=True)[n]  # Load the Target
        Solution = np.load('BestSol_seg.npy', allow_pickle=True)[n]  # Load the Best solution
        Pred, Eval = model_region_growing(Feat, sol=Solution[-1])
        Segmented.append(Pred)
    np.save('Segmented.npy', np.asarray(Segmented))

# Optimization For Feature Selection and for parameter optimization in ensemble techniques
an = 0
if an == 1:
    Fitness=[]
    BESTSOL=[]
    for n in range(no_of_dataset): # For datasets
        Feat = np.load('Segmented.npy', allow_pickle=True)[n]  # Load the Data
        Target = np.load('Target.npy', allow_pickle=True)[n]  # Load the Target
        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 13 # 10 for Feature selection, 1 for epoches in VGG16, 1 for activation function in Resnet, 1 for hidden neuron count in Inception
        xmin = matlib.repmat(np.concatenate([np.zeros((1, 10)), 50, 1, 5], axis=None),Npop, 1)
        xmax = matlib.repmat(np.concatenate([100 * np.ones((1, 10)), 100, 5, 255],axis=None), Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = uniform(xmin[i, j], xmax[i, j])
        fname = Objfun
        max_iter = 50

        print("POA...")
        [bestfit1, fitness1, bestsol1, time1] = POA(initsol, fname, xmin, xmax, Max_iter)  # POA

        print("HBA...")
        [bestfit2, fitness2, bestsol2, time2] = HBA(initsol, fname, xmin, xmax, Max_iter)  # HBA

        print("ZOA...")
        [bestfit3, fitness3, bestsol3, time3] = ZOA(initsol, fname, xmin, xmax, Max_iter)  # ZOA

        print("KHO...")
        [bestfit4, fitness4, bestsol4, time4] = KHO(initsol, fname, xmin, xmax, Max_iter)  # KHO

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

        BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

        Fitness.append(fitness)
        BESTSOL.append(BestSol_CLS)

    np.save('BESTSOL.npy', BESTSOL)
    np.save('Fitness.npy', Fitness)

# Feature Selection
an = 0
if an == 1:
    Selected_Feature = []
    for n in range(no_of_dataset): # For 3 datasets
        Feat = np.load('Segmented.npy', allow_pickle=True)[n]
        Target = np.load('Target.npy', allow_pickle=True)[n]
        BESTSOL = np.load('BESTSOL.npy', allow_pickle=True)[n][-1, :]
        Feature_VGG16, Eval_VGG16, Pred_VGG16 = Model_VGG16_Feat(Feat, Target, sol=BESTSOL[10])
        Feature_Resnet, Eval_Resnet, Pred_Resnet = Model_Resnet_Feat(Feat, Target, sol=BESTSOL[11])
        Feature_Inception, Eval_Inception, Pred_Inception = Model_Inception_Feat(Feat, Target, sol=BESTSOL[12])
        Feature_fusion = np.concatenate((Feature_VGG16, Feature_Resnet, Feature_Inception), axis=0)
        featureselection = Feature_fusion[:, int(BESTSOL[:10])]
        Selected_Feature.append(featureselection)
    np.save('Feature_Selected.npy', Selected_Feature)

# Optimization for Classification
an = 0
if an == 1:
    FITNESS = []
    BESTSOL = []
    for n in range(no_of_dataset):
        Feat = np.load('Feature_Selected.npy', allow_pickle=True)[n]  # Load the Feature Selected
        Target = np.load('Target.npy', allow_pickle=True)[n]  # Load the Target
        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden neuron count, Learning rate, Step per Epoch in GRU
        xmin = matlib.repmat(np.asarray([5, 0.01, 100]), Npop, 1)
        xmax = matlib.repmat(np.asarray([255, 0.99, 200]), Npop, 1)
        fname = objfun_cls
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("POA...")
        [bestfit1, fitness1, bestsol1, time1] = POA(initsol, fname, xmin, xmax, Max_iter)  # POA

        print("HBA...")
        [bestfit2, fitness2, bestsol2, time2] = HBA(initsol, fname, xmin, xmax, Max_iter)  # HBA

        print("ZOA...")
        [bestfit3, fitness3, bestsol3, time3] = ZOA(initsol, fname, xmin, xmax, Max_iter)  # ZOA

        print("KHO...")
        [bestfit4, fitness4, bestsol4, time4] = KHO(initsol, fname, xmin, xmax, Max_iter)  # KHO

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

        BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

        FITNESS.append(fitness)
        BESTSOL.append(BestSol_CLS)

    np.save('Fitness.npy', np.asarray(FITNESS))  # Save the FITNESS
    np.save('BestSol_cls.npy', np.asarray(BESTSOL))  # Save the Bestsol

# KFOLD - Classification
an = 0
if an == 1:
    EVAL_ALL = []
    for n in range(no_of_dataset):
        Feat = np.load('Feature_Selected.npy', allow_pickle=True)[n]  # Load the Images
        Target = np.load('Target.npy', allow_pickle=True)[n]  # Load the Target
        BestSol = np.load('BestSol_cls.npy', allow_pickle=True)[n]  # Load the Best Solution
        K = 5
        Per = 1 / 5
        Perc = round(Feat.shape[0] * Per)
        EVAL = []
        for i in range(K):
            Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Feat.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Feat[train_index, :]
            Train_Target = Target[train_index, :]
            Eval = np.zeros((10, 25))
            for j in range(BestSol.shape[0]):
                sol = BestSol[j, :]
                Eval[j, :], pred0 = Model_OELRC(Train_Data, Train_Target, Test_Data, Test_Target, sol=sol)
            Eval[5, :], pred_1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :], pred_2 = Model_DCN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :], pred_3 = Model_BILSTM(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :], pred_4 = Model_OELRC(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        EVAL_ALL.append(EVAL)
    np.save('Eval_Fold.npy', np.asarray(EVAL_ALL))

plot_Fitness()
Plot_ROC()
plot_results_kfold()
Image_Results()

