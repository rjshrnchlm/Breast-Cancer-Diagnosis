import numpy as np
from sklearn.linear_model import Ridge
from Evaluation import evaluation


def Model_OELRC(train_data, train_target, test_data, test_target, HN=None, sol=None, precision=None, max_iter=None):
    """
    Optimized Extreme Learning with Ridge Classifier (O-ELRC) Model

    Parameters from paper:
    - HN (Hidden Neuron Count for ELM): Range [5, 255] (default: 128)
    - precision (Tolerance for Ridge Classifier): Range [0.01, 0.99] (default: 0.01)
    - max_iter (Maximum Iterations for Ridge Classifier): Range [100, 200] (default: 150)
    """
    # 1. Set default values within the paper's specified ranges
    if sol is None:
        sol = [5, 0.01, 100]
    if HN is None:
        HN = int(sol[0])  # 128  # Range: [5, 255]
    if precision is None:
        precision = sol[1]  # Range: [0.01, 0.99]
    if max_iter is None:
        max_iter = int(sol[2])  # Range: [100, 200]

    # 2. Reshape / Flatten input data into 2D feature vectors (N, Features)
    Train_X = np.reshape(train_data, (train_data.shape[0], -1)).astype('float32')
    Test_X = np.reshape(test_data, (test_data.shape[0], -1)).astype('float32')

    # Normalize features to prevent numerical overflow in exponential layers
    Train_X = (Train_X - np.mean(Train_X, axis=0)) / (np.std(Train_X, axis=0) + 1e-8)
    Test_X = (Test_X - np.mean(Test_X, axis=0)) / (np.std(Test_X, axis=0) + 1e-8)

    # Part A: Extreme Learning Machine (ELM) Classifier
    np.random.seed(42)
    n_features = Train_X.shape[1]

    # Randomly initialize input weights (W) and biases (b)
    W = np.random.normal(loc=0.0, scale=0.5, size=(n_features, HN))
    b = np.random.normal(loc=0.0, scale=0.5, size=(1, HN))

    # Hidden layer activation matrix (Sigmoid)
    H_train = 1.0 / (1.0 + np.exp(- (np.dot(Train_X, W) + b)))
    H_test = 1.0 / (1.0 + np.exp(- (np.dot(Test_X, W) + b)))

    # Compute output weights using Moore-Penrose pseudo-inverse (Beta = H^+ * Y)
    beta = np.dot(np.linalg.pinv(H_train), train_target)

    # Classified diagnosis score 1 from ELM
    score_elm = np.dot(H_test, beta)

    # Part B: Ridge Classifier
    # Solves (X^T * X + alpha * I)^(-1) * X^T * Y using the specified precision and max_iter
    ridge_clf = Ridge(
        alpha=1.0,
        tol=precision,
        max_iter=max_iter,
        solver='sag'
    )
    ridge_clf.fit(Train_X, train_target)

    # Classified diagnosis score 2 from Ridge Classifier
    score_ridge = ridge_clf.predict(Test_X)

    # Part C: Ensemble Score Averaging & Final Prediction
    # Normalize classifier outputs to [0, 1] range before averaging
    def min_max_norm(score):
        return (score - np.min(score)) / (np.max(score) - np.min(score) + 1e-8)

    score_elm_norm = min_max_norm(score_elm)
    score_ridge_norm = min_max_norm(score_ridge)

    # Average the classifier scores (Section 3.1 & Figure 3)
    final_score = (score_elm_norm + score_ridge_norm) / 2.0

    # Decision thresholding
    pred = np.zeros_like(final_score)
    avg = np.mean(final_score)
    pred[final_score >= avg] = 1
    pred[final_score < avg] = 0
    pred = pred.astype('int')

    # Performance Evaluation
    Eval = evaluation(test_target, pred)
    return Eval, pred