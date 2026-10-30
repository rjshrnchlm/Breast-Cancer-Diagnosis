import time
import numpy as np
import matplotlib.pyplot as plt


def findlimits(ns, lb, ub, best, rng):
    """
    Evolutionary Boundary Constraint Handling Scheme.

    Equivalent to the MATLAB findlimits() function.
    """
    ns = np.asarray(ns, dtype=float).copy()
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    best = np.asarray(best, dtype=float)

    below = ns < lb
    above = ns > ub

    # Correct values below lower bounds
    if np.any(below):
        a = rng.random()
        ns[below] = a * lb[below] + (1.0 - a) * best[below]

    # Correct values above upper bounds
    if np.any(above):
        b = rng.random()
        ns[above] = b * ub[above] + (1.0 - b) * best[above]

    return ns


def KHO(X, cost, UB, LB, MI):
    NP, NK = X.shape
    NR = 10       # Number of runs
    # NK = 25       # Number of krills
    # MI = 200      # Maximum iterations
    C_flag = 1    # Crossover flag: 1 = enabled

    # Search-space bounds
    # NP = 10
    # UB = 10.0 * np.ones(NP)
    # LB = -10.0 * np.ones(NP)

    Dt = np.mean(np.abs(UB[0] - LB[0])) / 2.0

    Vf = 0.02
    Dmax = 0.005
    Nmax = 0.01

    # Random-number generator
    rng = np.random.default_rng()

    # Best fitness history: iteration 0 through MI
    Kgb = np.zeros((MI + 1, NR))

    # Best position history for every run
    Xgb = np.zeros((NP, MI + 1, NR))
    ct = time.time()
    convergence = np.zeros(MI + 1)
    # Optimization and Simulation
    for nr in range(NR):

        # # Initialize krill positions
        # X = LB[:, None] + (
        #     UB - LB
        # )[:, None] * rng.random((NP, NK))

        # Evaluate initial fitness
        K = np.array([cost(X[:, i]) for i in range(NK)])

        # Individual best fitness and positions
        Kib = K.copy()
        Xib = X.copy()

        # Global best
        A = np.argmin(K)
        Kgb[0, nr] = K[A]
        Xgb[:, 0, nr] = X[:, A]

        # Initialize motion matrices
        F = np.zeros((NP, NK))
        N = np.zeros((NP, NK))

        # --------------------------------------------------
        # Main iteration loop
        # --------------------------------------------------
        for j in range(1, MI + 1):

            best_position = Xgb[:, j - 1, nr].copy()
            best_fitness = Kgb[j - 1, nr]

            # ----------------------------------------------
            # Virtual Food Location
            # ----------------------------------------------
            # MATLAB:
            # Sf(ll) = sum(X(ll,:)./K)
            #
            # Xf = Sf / sum(1./K)
            #
            # Handle zero fitness values safely.
            if np.any(K == 0):
                zero_indices = np.where(K == 0)[0]
                food_position = np.mean(
                    X[:, zero_indices], axis=1
                )
            else:
                with np.errstate(divide="ignore",
                                 invalid="ignore"):
                    weights = 1.0 / K
                    denominator = np.sum(weights)

                    if (
                        not np.isfinite(denominator)
                        or abs(denominator) < 1e-15
                    ):
                        food_position = np.mean(X, axis=1)
                    else:
                        food_position = (
                            X @ weights
                        ) / denominator

            # food_position = findlimits(
            #     food_position, LB[j-1], UB[j-1],
            #     best_position, rng
            # )

            # food_position = np.clip(food_position, LB, UB)
            food_position = np.clip(food_position, LB[:,0], UB[:, 0])

            Kf = cost(food_position)

            # Preserve the best virtual food location
            # encountered so far.
            if j > 1:
                if Kf > previous_food_fitness:
                    food_position = previous_food_position.copy()
                    Kf = previous_food_fitness

            previous_food_position = food_position.copy()
            previous_food_fitness = Kf

            # Movement weight
            w = 0.1 + 0.8 * (1.0 - j / MI)

            # Fitness range used for normalization
            Kw_Kgb = np.max(K) - best_fitness

            # Prevent division by zero in degenerate cases
            if abs(Kw_Kgb) < 1e-15:
                Kw_Kgb = 1e-15

            # ----------------------------------------------
            # Process each krill
            # ----------------------------------------------
            for i in range(NK):

                # Distances from krill i to food and best
                Rf = food_position - X[:, i]
                Rgb = best_position - X[:, i]

                # Distances to all krill
                RR = X - X[:, i:i + 1]
                R = np.sqrt(np.sum(RR * RR, axis=0))

                # ==========================================
                # 1. Movement Induced by Best Krill
                # ==========================================
                distance_best = np.linalg.norm(Rgb)

                if Kgb[j - 1, nr] < K[i] and distance_best > 1e-15:
                    alpha_b = (
                        -2.0
                        * (1.0 + rng.random() * (j / MI))
                        * (Kgb[j - 1, nr] - K[i])
                        / Kw_Kgb
                        / distance_best
                        * Rgb
                    )
                else:
                    alpha_b = np.zeros(NP)

                # ==========================================
                # 2. Movement Induced by Neighbors
                # ==========================================
                nn = 0
                ds = np.mean(R) / 5.0
                alpha_n = np.zeros(NP)

                for n in range(NK):
                    if R[n] < ds and n != i:
                        nn += 1

                        if nn <= 4 and K[i] != K[n]:
                            if R[n] > 1e-15:
                                alpha_n -= (
                                    (K[n] - K[i])
                                    / Kw_Kgb
                                    / R[n]
                                    * RR[:, n]
                                )

                # Induced movement
                N[:, i] = (
                    w * N[:, i]
                    + Nmax * (alpha_b + alpha_n)
                )

                # ==========================================
                # 3. Foraging Motion: Food Attraction
                # ==========================================
                distance_food = np.linalg.norm(Rf)

                if Kf < K[i] and distance_food > 1e-15:
                    Beta_f = (
                        -2.0
                        * (1.0 - j / MI)
                        * (Kf - K[i])
                        / Kw_Kgb
                        / distance_food
                        * Rf
                    )
                else:
                    Beta_f = np.zeros(NP)

                # Attraction toward individual best position
                Rib = Xib[:, i] - X[:, i]
                distance_individual_best = np.linalg.norm(Rib)

                if (
                    Kib[i] < K[i]
                    and distance_individual_best > 1e-15
                ):
                    Beta_b = (
                        -(Kib[i] - K[i])
                        / Kw_Kgb
                        / distance_individual_best
                        * Rib
                    )
                else:
                    Beta_b = np.zeros(NP)

                # Foraging motion
                F[:, i] = (
                    w * F[:, i]
                    + Vf * (Beta_b + Beta_f)
                )

                # ==========================================
                # 4. Physical Diffusion
                # ==========================================
                diffusion = (
                    Dmax
                    * (1.0 - j / MI)
                    * float(
                        rng.random()
                        + (K[i] - Kgb[j - 1, nr])
                        / Kw_Kgb
                        >= 1.0
                    )
                    * (2.0 * rng.random(NP) - 1.0)
                )

                # ==========================================
                # 5. Motion Process
                # ==========================================
                # The supplied MATLAB code calculates D but
                # does not add it to DX. This preserves that
                # behavior.
                DX = Dt * (N[:, i] + F[:, i])

                # ==========================================
                # 6. Crossover
                # ==========================================
                if C_flag == 1:
                    C_rate = (
                        0.8
                        + 0.2
                        * (K[i] - Kgb[j - 1, nr])
                        / Kw_Kgb
                    )

                    Cr = rng.random(NP) < C_rate

                    # Select a random krill for crossover
                    NK4Cr = rng.integers(0, NK)

                    X[:, i] = (
                        X[:, NK4Cr] * (~Cr)
                        + X[:, i] * Cr
                    )

                # ==========================================
                # 7. Update Position and Fitness
                # ==========================================
                X[:, i] = X[:, i] + DX

                # X[:, i] = findlimits(
                #     X[:, i], LB[j-1], UB[j-1],
                #     best_position, rng
                # )

                X[:, i] = np.clip(X[:, i], LB[:, i], UB[:, i])
                K[i] = cost(X[:, i])

                # Update individual best
                if K[i] < Kib[i]:
                    Kib[i] = K[i]
                    Xib[:, i] = X[:, i].copy()

                convergence[j] = best_fitness
            # ----------------------------------------------
            # Update Global Best
            # ----------------------------------------------
            A = np.argmin(K)

            if K[A] < Kgb[j - 1, nr]:
                Kgb[j, nr] = K[A]
                Xgb[:, j, nr] = X[:, A].copy()
            else:
                Kgb[j, nr] = Kgb[j - 1, nr]
                Xgb[:, j, nr] = Xgb[:, j - 1, nr].copy()

        print(
            f"Run {nr + 1:2d}/{NR}: "
            f"Best fitness = {Kgb[-1, nr]:.10g}"
        )
    # --------------------------------------------------
    # Post-processing
    # --------------------------------------------------
    Ron_No = np.argmin(Kgb[-1, :])
    Best = Kgb[-1, Ron_No]
    # Best_position = Xgb[:, -1, Ron_No]
    Best_position = Xib[Ron_No]
    Mean = np.mean(Kgb[-1, :])
    Worst = np.max(Kgb[-1, :])
    Standard_Deviation = np.std(Kgb[-1, :], ddof=1)

    # Convergence Plot
    iterations = np.arange(MI + 1)
    best_run_values = Kgb[:, Ron_No]
    average_run_values = np.mean(Kgb, axis=1)

    # semilogy requires positive values
    best_plot = np.maximum(best_run_values, 1e-15)
    average_plot = np.maximum(average_run_values, 1e-15)
    convergence = Kgb[:-1,0]
    ct = time.time() - ct
    return Best, convergence, Best_position, ct

