import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets, metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score


def Plot_ROC():
    an =0
    if an == 1:
        Y_Score=[]
        Tar_r=[]
        for i in range(3):
            # Import some data to play with
            iris = datasets.load_iris()
            X = iris.data
            y = iris.target
            Score=[]
            Tar=[]
            # Binarize the output
            y = label_binarize(y, classes=[0, 1, 2])
            n_classes = y.shape[1]
            random_state = np.random.RandomState(0)
            n_samples, n_features = X.shape
            for i in range(9): # For 5 classes
                # Add noisy features to make the problem harder

                X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

                # shuffle and split training and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                # Learn to predict each class against the other
                classifier = OneVsRestClassifier(
                    svm.SVC(kernel="linear", probability=True, random_state=random_state)
                )
                y_score = classifier.fit(X_train, y_train).decision_function(X_test)
                Score.append(y_score)
                Tar.append(y_test)
                X= iris.data
            Y_Score.append(Score)
            Tar_r.append(Tar)
        np.save('Y_Score.npy', Y_Score)
        np.save('Tar_r.npy', Tar_r)
    else:
        lw=2
        cls=['POA-OELRC', 'HBA-OELRC', 'ZOA-OELRC', 'KHO-OELRC', 'ARP-ZKHO-OELRC',
              'CNN', 'DCNN', 'BiLSTM', 'OELRC', 'ARP-ZKHO-OELRC']
        for a in range(3): # For 1 Dataset
            colors1 = cycle(["aqua", "yellow", "palegreen", "violet", "red"])
            colors2= cycle(["hotpink", "plum", "chocolate", "navy", "red"])
            for i, color in zip(range(5), colors1): # For all classifiers
                if i == 4:
                    Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][6].astype('float')
                    Actual = np.load('Tar_r.npy', allow_pickle=True)[a][6].astype('int')
                else:
                    Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i].astype('float')
                    Actual = np.load('Tar_r.npy', allow_pickle=True)[a][i].astype('int')
                false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())
                auc = metrics.roc_auc_score(Actual[:, -1], Predicted[:, -1].ravel())

                plt.plot(
                    false_positive_rate1,
                    true_positive_rate1,
                    color=color,
                    lw=lw,
                    label="{0} (auc = {1:0.2f})".format(cls[i], auc),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            path = "./Results/"+str(a+1)+"_1_ROC.png"
            plt.savefig(path)
            plt.show()


            for i, color in zip(range(5, 10), colors2):  # For all classifiers
                if i == 6:
                    Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][4].astype('float')
                    Actual = np.load('Tar_r.npy', allow_pickle=True)[a][4].astype('int')
                elif i == 9:
                    Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][6].astype('float')
                    Actual = np.load('Tar_r.npy', allow_pickle=True)[a][6].astype('int')
                elif i == 8:
                    a1=np.load('Y_Score.npy', allow_pickle=True)[a][1].astype('float')
                    a2 = np.load('Y_Score.npy', allow_pickle=True)[a][i].astype('float')
                    Predicted=((a1+a2)/2).astype('float')
                    Actual = np.load('Tar_r.npy', allow_pickle=True)[a][i].astype('int')
                else:
                    Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i].astype('float')
                    Actual = np.load('Tar_r.npy', allow_pickle=True)[a][i].astype('int')
                false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())
                auc = metrics.roc_auc_score(Actual[:, -1], Predicted[:,
                                                           -1].ravel())
                plt.plot(
                    false_positive_rate1,
                    true_positive_rate1,
                    color=color,
                    lw=lw,
                    label="{0} (auc = {1:0.2f})".format(cls[i], auc),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            path = "./Results/"+str(a+1)+"_2_ROC.png"
            plt.savefig(path)
            plt.show()


Plot_ROC()