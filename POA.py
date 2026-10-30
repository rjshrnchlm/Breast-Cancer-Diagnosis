import numpy as np
import time


# Pelican Optimization Algorithm (POA)
def POA(X,fitness , lowerbound, upperbound, Max_iterations):
    SearchAgents, dimension = X.shape

    fit = np.zeros((SearchAgents))
    for i in range(SearchAgents):
        fit[i] = fitness(X[i, :])
    Convergence = np.zeros(Max_iterations)
    ct = time.time()

    for t in range(Max_iterations):
        # update the best condidate solution
        best, location = np.min(fit), np.argmin(fit)

        if t == 0:
            Xbest = X[location, :]
            fbest = best
        else:
            if best < fbest:
                fbest = best
                Xbest = X[location, :]
        # UPDATE location of food
        X_FOOD = []
        k = np.random.permutation(SearchAgents)
        X_FOOD = X[k, :]
        F_FOOD = fit[k]

        for i in range(SearchAgents):
            # PHASE 1: Moving towards prey (exploration phase)
            I = np.round(1 + np.random.rand(1, 1))
            if fit[i] > F_FOOD[i]:
                X_new = X[i, :] + np.multiply(np.random.rand(1, 1), (X_FOOD - np.multiply(I, X[i, :])))
            else:
                X_new = X[i, :] + np.multiply(np.random.rand(1, 1), (X[i, :] - 1.0 * X_FOOD))

            f_new = fitness(X_new[i])
            if f_new <= fit[i]:
                X[i, :] = X_new[i, :]
                fit[i] = f_new
            # END PHASE 1: Moving towards prey (exploration phase)
            # PHASE 2: Winging on the water surface (exploitation phase)
            X_new[i] = X[i, :] + np.multiply(
                np.multiply(0.2 * (1 - t / Max_iterations), (2 * np.random.rand(1, dimension) - 1)), X[i, :])

            # Updating X_i using (7)
            f_new = fitness(X_new[i])
            if f_new <= fit[i]:
                X[i, :] = X_new[i]
                fit[i] = f_new
            # END PHASE 2: Winging on the water surface (exploitation phase)
            # Boundary conditions
            X_new[i, :] = np.clip(X_new[i, :], lowerbound[i], upperbound[i])

        Convergence[t] = fbest
    CT = time.time() - ct
    Best_score = fbest
    Best_pos = Xbest
    return Best_score, Convergence, Best_pos, CT

