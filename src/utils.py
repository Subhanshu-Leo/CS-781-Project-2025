"""
Utility functions for verification project
"""

import numpy as np


def is_positive_definite(M):
    """Check if matrix is positive definite"""
    eigenvalues = np.linalg.eigvals(M)
    is_symmetric = np.allclose(M, M.T)
    all_positive = np.all(eigenvalues > 1e-10)
    return is_symmetric and all_positive


def check_controllability(A, B):
    """Check Kalman controllability condition"""
    n = A.shape[0]
    C = B.copy()
    for i in range(1, n):
        C = np.hstack([C, np.linalg.matrix_power(A, i) @ B])
    rank = np.linalg.matrix_rank(C)
    return rank == n


def check_observability(A, C):
    """Check observability"""
    n = A.shape[0]
    O = C.copy()
    for i in range(1, n):
        O = np.vstack([O, C @ np.linalg.matrix_power(A, i)])
    rank = np.linalg.matrix_rank(O)
    return rank == n


def load_learned_parameters(filepath='results/learned_parameters.json'):
    """
    Load learned parameters from Phase 1

    Returns:
        Q, R, K matrices and violation_rate
    """
    import json

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            learned_params = data['best_params']

            Q = np.array(learned_params['Q'])
            R = np.array(learned_params['R'])
            K = np.array(learned_params['K'])
            violation_rate = learned_params['violation_rate']

            return Q, R, K, violation_rate
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Learned parameters not found at {filepath}. "
            "Please run Phase 1 first: python src/1_learn_parameters.py"
        )


def compute_lqr_gain(A, B, Q, R):
    """
    Compute LQR gain from Q, R matrices

    Returns:
        K gain matrix, or None if unstable
    """
    from scipy.linalg import solve_continuous_are

    try:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P

        # Verify stability
        A_cl = A - B @ K
        eigenvalues = np.linalg.eigvals(A_cl)

        if np.all(np.real(eigenvalues) < 0):
            return K
        else:
            return None
    except Exception:
        return None


def wilson_score_interval(successes, trials, confidence=0.95):
    """
    Compute Wilson score confidence interval for proportions

    Args:
        successes: Number of successes
        trials: Total number of trials
        confidence: Confidence level (default 0.95)

    Returns:
        (lower, upper) bounds
    """
    from scipy import stats

    if trials == 0:
        return 0.0, 0.0

    z = stats.norm.ppf((1 + confidence) / 2)
    p_hat = successes / trials

    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(
        p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)
    ) / denominator

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return lower, upper
