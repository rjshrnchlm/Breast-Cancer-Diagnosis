import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable



def plot_Fitness():
    for a in range(3): # For 3 datasets
        fitness = np.load('Fitness.npy', allow_pickle=True)[a]
        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = [ 'HFA', 'SWOA', 'RHS', 'DA', 'PROPOSED']
        # for i in range(fitness.shape[0]):
        fit = fitness
        ind = np.argsort(fit[:, fit.shape[1] - 1])  # [:, fit.shape[1] - 1]
        x = fit[ind[0], :].copy()
        y = fit[4, :].copy()
        fit[ind[0], :] = y.copy()
        fit[4, :] = x.copy()
        x = fit[ind[1], :].copy()
        y = fit[2, :].copy()
        fit[ind[1], :] = y.copy()
        fit[2, :] = x.copy()

        Value = np.zeros((fit.shape[0], 5))
        for j in range(fit.shape[0]):
            Value[j, 0] = np.min(fit[j, :])
            Value[j, 1] = np.max(fit[j, :])
            Value[j, 2] = np.mean(fit[j, :])
            Value[j, 3] = np.median(fit[j, :])
            Value[j, 4] = np.std(fit[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print(
            '-------------------------------------------------- Statistical Analysis Dataset - ',str(a+1),' --------------------------------------------------')
        print(Table)

        iteration = np.arange(fit.shape[1])
        plt.plot(iteration, fit[0, :], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                 label="POA-OELRC")
        plt.plot(iteration, fit[1, :], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                 label="HBA-OELRC")
        plt.plot(iteration, fit[2, :], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                 label="ZOA-OELRC")
        plt.plot(iteration, fit[3, :], color='m', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                 label="KHO-OELRC")
        plt.plot(iteration, fit[4, :], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                 label="ARP-ZKHO-OELRC")
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=4)
        path1 = "./Results/convergence-"+str(a+1)+".png"
        plt.savefig(path1)
        plt.show()


def plot_results_kfold():
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 3, 8]

    Algorithm = ['TERMS', 'POA', 'HBA', 'ZOA', 'KHO', 'PROPOSED'] # POA - Pelican Optimization Algorithm
    Classifier = ['TERMS', 'Ref-1', 'Ref-5', 'Ref-6', 'Without Opt', 'PROPOSED']
    Dataset = ['DATASET-1', 'DATASET-2', 'DATASET-3']

    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    learnper = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]
        value[:, :-1] = value[:, :-1] * 100
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('-------------------------------------------------- ', Dataset[i], ' - Algorithm Comparison - ',
              'Learning Percentage --------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- ', Dataset[i], '-Classifier Comparison - ',
              'Learning Percentage --------------------------------------------------')
        print(Table)

        eval = np.load('Eval_Fold.npy', allow_pickle=True)
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            # Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if Graph_Term[j] == 9:
                        Graph[k, l] = eval[i,k, l, Graph_Term[j]+4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100

            plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='*', markerfacecolor='blue', markersize=16,
                     label="POA-OELRC")
            plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='*', markerfacecolor='red', markersize=16,
                     label="HBA-OELRC")
            plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='*', markerfacecolor='green', markersize=16,
                     label="ZOA-OELRC")
            plt.plot(learnper, Graph[:, 3], color='m', linewidth=3, marker='*', markerfacecolor='yellow', markersize=16,
                     label="KHO-OELRC")
            plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='*', markerfacecolor='cyan', markersize=16,
                     label="ARP-ZKHO-OELRC")
            plt.xlabel('5 - Fold')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=4)
            path1 = "./Results/Dataset_%s_%s_line_2.png" % (i+1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="CNN")
            ax.bar(X + 0.10, Graph[:, 6], color='#cc9f3f', width=0.10, label="DCNN")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="BiLSTM")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="OELRC")
            ax.bar(X + 0.40, Graph[:, 9], color='g', width=0.10, label="ARP-ZKHO-OELRC")

            plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
            plt.xlabel('5 - Fold')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_%s_%s_bar_2.png" % (i+1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()



plot_Fitness()
plot_results_kfold()



