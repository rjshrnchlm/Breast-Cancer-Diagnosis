import time
import numpy as np


# Honey Badger optimization algorithm (HBA)
def HBA(X, objfunc, lb, ub, tmax):
    N, dim = X.shape
    beta = 6  # the ability of HB to get the food (Eq. 4)
    C = 2  # constant in Eq. (3)
    vec_flag = np.array([1, -1])

    # Evaluation
    fitness = fun_calcobjfunc(objfunc, X)
    GYbest = np.min(fitness)
    gbest = np.argmin(fitness)
    Xprey = X[gbest, :].copy()

    CNVG = np.zeros(tmax)

    # Main loop
    for t in range(tmax):
        alpha = C * np.exp(-t / tmax)  # density factor in Eq. (3)
        I = Intensity(N, Xprey, X)  # intensity in Eq. (2)

        for i in range(N):
            r = np.random.rand()
            # r = (fitness[i] + np.min(fitness)) / (np.max(fitness) + np.mean(fitness) + fitness[i])
            F = vec_flag[int(np.floor(2 * np.random.rand()))]

            Xnew = X[i, :].copy()
            for j in range(dim):
                di = Xprey[j] - X[i, j]

                if r < 0.5:
                    r3, r4, r5 = np.random.rand(), np.random.rand(), np.random.rand()
                    Xnew[j] = Xprey[j] + F * beta * I[i] * Xprey[j] + F * r3 * alpha * di * abs(
                        np.cos(2 * np.pi * r4) * (1 - np.cos(2 * np.pi * r5)))
                else:
                    r7 = np.random.rand()
                    Xnew[j] = Xprey[j] + F * r7 * alpha * di

            Xnew = np.clip(Xnew, lb, ub)

            tempFitness = objfunc(Xnew)
            if tempFitness[i] < fitness[i]:
                fitness[i] = tempFitness[i]
                X[i, :] = Xnew[i]

        X = np.clip(X, lb, ub)
        Ybest = np.min(fitness)
        CNVG[t] = Ybest

        if Ybest < GYbest:
            GYbest = Ybest
            Xprey = X[np.argmin(fitness), :].copy()

    Food_Score = GYbest
    ct = time.time()
    return Food_Score, CNVG, Xprey, ct


def fun_calcobjfunc(func, X):
    N = X.shape[0]
    Y = np.zeros(N)
    for i in range(N):
        Y[i] = func(X[i, :])
    return Y


def Intensity(N, Xprey, X):
    di = np.zeros(N)
    S = np.zeros(N)

    for i in range(N - 1):
        di[i] = (np.linalg.norm(X[i, :] - Xprey) + np.finfo(float).eps) ** 2
        S[i] = (np.linalg.norm(X[i, :] - X[i + 1, :]) + np.finfo(float).eps) ** 2

    di[N - 1] = (np.linalg.norm(X[N - 1, :] - Xprey) + np.finfo(float).eps) ** 2
    S[N - 1] = (np.linalg.norm(X[N - 1, :] - X[0, :]) + np.finfo(float).eps) ** 2

    I = np.zeros(N)
    for i in range(N):
        r2 = np.random.rand()
        I[i] = r2 * S[i] / (4 * np.pi * di[i])

    return I
