import numpy as np
import pandas as pd
from typing import Optional, Tuple, TypeAlias

# Type alias for the function return type f_val, gradient, hessian
FunctionEvaluationResult: TypeAlias = Tuple[float, np.ndarray, Optional[np.ndarray]]


# Load the COV matrix from a CSV file
cov_matrix_path = 'strategy_portfolio_opt/covariance_matrix.csv'
cov_mat_df = pd.read_csv(cov_matrix_path, index_col=0)
cov_mat = cov_mat_df.values

# Load the expected returns from a CSV file
expected_returns_path = 'strategy_portfolio_opt/mean_pnls.json'
expected_returns_dict = pd.read_json(expected_returns_path, orient='index')
expected_returns = expected_returns_dict.values.flatten()

# Asset the indexes for the covariance matrix and expected returns
asset_indexes = cov_mat_df.index.to_list()
if not np.array_equal(asset_indexes, expected_returns_dict.index.to_list()):
    raise ValueError("Asset indexes in covariance matrix and expected returns do not match.")

# --- Test Asset Allocation Problem ---
def f_aa(w: np.ndarray, eval_hessian: bool = True) -> FunctionEvaluationResult:
    """
    Asset Allocation function for testing.
    
    Parameters
    ----------
    w : np.ndarray
        Input vector.
    eval_hessian : bool, optional
        Whether to evaluate the Hessian. Default is True.
    Returns
    -------
    FunctionEvaluationResult
        Tuple containing function value, gradient, and Hessian (if eval_hessian is True).
    """
    f_val = w.T @ cov_mat @ w
    grad = 2 * cov_mat @ w
    hess = 2 * cov_mat if eval_hessian else None
    return f_val, grad, hess

# Define the equality constraints for the asset allocation problem: sum of weights equals 1
eq_constraints_mat_aa = np.ones((1, cov_mat.shape[0]))
eq_constraints_rhs_aa = np.array([1.0])

# Solver expectation is g(x) <= 0
# Define the inequality constraints for the asset allocation problem: expected returns >= 5
zero_matrix = np.zeros_like(cov_mat)

ineq_aa = [
    lambda w, eval_hessian=True: (
        0.05 - expected_returns @ w,   # g(w) ≤ 0  ⇔  μᵀw ≥ 0.05
        -expected_returns,
        zero_matrix
    )
]

# ─── long-only:  w_i ≥ 0   ⇒   g_i(w) = -w_i ≤ 0  for every i ─────────
neg_eye = -np.eye(cov_mat.shape[0])                   # store gradients once

def make_nonneg(i: int) -> callable:
    g_vec = np.zeros(cov_mat.shape[0]);  g_vec[i] = -1      # ∇g_i = -e_i
    def g_i(w, *, eval_hessian=True):
        return -w[i], g_vec, zero_matrix           # g_i(w) ≤ 0  ⇔ w_i ≥ 0
    return g_i

for i in range(cov_mat.shape[0]):
    ineq_aa.append(make_nonneg(i))