"""
Sparse logistic regression via ADMM
===================================

  min_x  f(x) + g(x)     where
  f(x) = (1/m) Σ log(1 + exp(-b_i a_i^T x)) + λ‖x‖_2²
  g(x) = μ‖x‖_1

We split x = z and solve with the scaled-form ADMM:

  x^{k+1} = argmin_x  f(x) + (ρ/2)‖x - z^k + u^k‖_2²      (Newton)
  z^{k+1} = S_{μ/ρ}(x^{k+1}+u^k)                           (soft-threshold)
  u^{k+1} = u^k + x^{k+1} - z^{k+1}

Stopping rule:  ‖r^k‖₂ = ‖x^k - z^k‖₂  and  ‖s^k‖₂ = ρ‖z^k - z^{k-1}‖₂
both ≤ 1e-6  *or*  k ≥ 10000.

Plots:
  (1) log10(‖r^k‖₂)  &  log10(‖s^k‖₂)  vs. iteration
  (2) log10(|ℓ(x^k) - ℓ(x*)|)          vs. iteration      (x* ≈ very-tight ADMM)

Statistics:
  for μ in {1e-3, 0.01, 0.05} report (#iters , sparsity).

Author: ChatGPT-o3, 2025-05-21
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
from sklearn.datasets import load_svmlight_file


# ---------------------------------------------------------------------
# I.  Data utilities
# ---------------------------------------------------------------------
def load_a9a(path: str | Path | None = None, add_intercept: bool = True):
    """
    Load the a9a (Adult) data-set.
    If *path* is None the routine tries to download automatically
    via scikit-learn's OpenML mirror. Otherwise *path* should point
    to a LibSVM-formatted file (train + test concatenated is fine).

    Returns
    -------
    A : csr_matrix, shape (m, n)
    b : ndarray,     shape (m,) in {-1, +1}
    """
    if path is None or not Path(path).exists():
        # fallback to OpenML (≈ 2 MB); comment out if offline
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("a9a", version=1, as_frame=False, parser="auto")
        A = sp.csr_matrix(X)
        b = y.astype(int)
        b[b == 0] = -1
    else:
        A, y = load_svmlight_file(path)
        b = y.astype(int)
        b[b == 0] = -1
    if add_intercept:
        m = A.shape[0]
        intercept = sp.csr_matrix(np.ones((m, 1)))
        A = sp.hstack([A, intercept], format="csr")
    return A, b


# ---------------------------------------------------------------------
# II.  Objective, gradient, Hessian helpers
# ---------------------------------------------------------------------
def logistic_loss_and_grad_hess(x, A, b, lam):
    """
    Compute f(x), ∇f(x), and Hessian H at x for the smooth part
        f(x) = (1/m) Σ log(1 + exp(-b_i a_i^T x)) + λ‖x‖²
    """
    m = A.shape[0]
    Ax = A @ x            # shape (m,)
    yAx = b * Ax
    # σ(-yAx) = 1 / (1 + exp(yAx))
    sigma = 1.0 / (1.0 + np.exp(yAx))
    f = (1.0 / m) * np.sum(np.log1p(np.exp(-yAx))) + lam * np.dot(x, x)

    # gradient
    g_factor = -b * sigma  # shape (m,)
    g = (1.0 / m) * (A.T @ g_factor) + 2 * lam * x  # shape (n,)

    # Hessian: (1/m) A^T diag(σ (1-σ)) A + 2λ I

    # exploit sparsity with element-wise mult

    s = sigma * (1 - sigma)  # shape (m,)
    SA = A.multiply(s[:, None])  # row-scaling, stays sparse
    H = (A.T @ SA) / m + 2 * lam * sp.eye(A.shape[1], format="csr")
    return f, g, H


# ---------------------------------------------------------------------
# III.  Newton with backtracking for the x-subproblem
# ---------------------------------------------------------------------
def newton_subproblem(x0, z, u, A, b, lam, rho,
                      tol=1e-8, max_iter=50, alpha=0.01, beta=0.5):
    """
    Solve   min_x  f(x) + (ρ/2)‖x - z + u‖²   by Newton.
    Returns new x and number of Newton iterations used.
    """
    x = x0.copy()
    for it in range(max_iter):
        f, g, H = logistic_loss_and_grad_hess(x, A, b, lam)
        g += rho * (x - z + u)
        H = H + rho * sp.eye(H.shape[0])

        # Newton direction  H p = -g
        p = spla.spsolve(H.tocsc(), -g)

        # Check Newton decrement
        newton_dec = np.dot(p, -g)
        if newton_dec / 2 <= tol:
            break

        # Backtracking line-search
        t = 1.0
        fx = f + (rho / 2) * np.linalg.norm(x - z + u) ** 2
        while True:
            x_new = x + t * p
            f_new, _, _ = logistic_loss_and_grad_hess(x_new, A, b, lam)
            fx_new = f_new + (rho / 2) * np.linalg.norm(x_new - z + u) ** 2
            if fx_new <= fx + alpha * t * newton_dec:
                break
            t *= beta
        x += t * p
    return x, it + 1


# ---------------------------------------------------------------------
# IV.  ADMM driver
# ---------------------------------------------------------------------
def admm_logreg(A, b, lam, mu, rho=1.0, x0=None,
                rtol=1e-6, stol=1e-6, max_iter=10000,
                newton_params=None, verbose=False):
    """
    ADMM for ℓ₂+ℓ₁ sparse logistic regression.

    Returns
    -------
    hist : dict with keys
        'r_norm', 's_norm', 'eps_pri', 'eps_dual',
        'obj', 'time', 'iters'
    x    : final primal variable
    """
    m, n = A.shape
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    z = x.copy()
    u = np.zeros_like(x)

    r_hist, s_hist, obj_hist, t_hist = [], [], [], []
    tic = time()
    np_soft = np.vectorize(lambda y, t: np.sign(y) * max(abs(y) - t, 0.0))

    params = dict(tol=1e-8, max_iter=50, alpha=0.01, beta=0.5)
    if newton_params:
        params.update(newton_params)

    for k in range(1, max_iter + 1):
        # x-update (Newton)
        x, _ = newton_subproblem(x, z, u, A, b, lam, rho, **params)

        # z-update (soft threshold)
        z_old = z.copy()
        x_hat = x + u
        z = np_soft(x_hat, mu / rho)

        # dual update
        u += x - z

        # diagnostics
        r = x - z
        s = rho * (z - z_old)
        r_norm = np.linalg.norm(r)
        s_norm = np.linalg.norm(s)

        # objective value
        f, _, _ = logistic_loss_and_grad_hess(x, A, b, lam)
        obj = f + mu * np.linalg.norm(x, 1)

        r_hist.append(r_norm)
        s_hist.append(s_norm)
        obj_hist.append(obj)
        t_hist.append(time() - tic)

        if verbose and k % 50 == 0:
            print(f"k={k:5d}  r={r_norm:.2e}  s={s_norm:.2e}  obj={obj:.4f}")

        # stopping
        if r_norm < rtol and s_norm < stol:
            break

    hist = dict(r_norm=np.array(r_hist),
                s_norm=np.array(s_hist),
                obj=np.array(obj_hist),
                time=np.array(t_hist),
                iters=k)
    return x, hist


# ---------------------------------------------------------------------
# V.  Experiment wrapper
# ---------------------------------------------------------------------
def run_experiment(mu_list=(1e-2,), lam=None, rho=1.0,
                   newton_params=None, data_path=None):
    # 1. load data
    A, b = load_a9a(data_path)
    m, n = A.shape
    if lam is None:
        lam = 1.0 / (2 * m)

    print(f"Data loaded: m={m:,}  n={n}  λ={lam:.3g}")

    results = {}
    # 2. high-accuracy solution (μ = first element, tight tol)
    print("\nComputing high-accuracy reference ...")
    x_star, _ = admm_logreg(A, b, lam, mu_list[0], rho,
                            rtol=1e-10, stol=1e-10, max_iter=20000,
                            newton_params=newton_params, verbose=False)
    l_star, _, _ = logistic_loss_and_grad_hess(x_star, A, b, lam)
    l_star += mu_list[0] * np.linalg.norm(x_star, 1)

    # 3. iterate over μ
    for mu in mu_list:
        print(f"\n>>> μ = {mu:.3g}")
        x, hist = admm_logreg(A, b, lam, mu, rho,
                              newton_params=newton_params,
                              verbose=True)
        sparsity = np.mean(np.abs(x) < 1e-10)
        results[mu] = dict(x=x, hist=hist, sparsity=sparsity)

        # plot (1) residuals
        k = np.arange(1, hist['iters'] + 1)
        plt.figure(1)
        plt.semilogy(k, hist['r_norm'], label=f"r, μ={mu}")
        plt.semilogy(k, hist['s_norm'], '--', label=f"s, μ={mu}")

        # plot (2) objective gap
        obj_gap = np.abs(hist['obj'] - l_star)
        plt.figure(2)
        plt.semilogy(k, obj_gap, label=f"μ={mu}")

    # finalise plots
    plt.figure(1)
    plt.xlabel("Iteration k")
    plt.ylabel("‖residual‖₂")
    plt.yscale("log")
    plt.legend()
    plt.title("ADMM optimality residuals")

    plt.figure(2)
    plt.xlabel("Iteration k")
    plt.ylabel("|ℓ(xᵏ) − ℓ(x*)|")
    plt.yscale("log")
    plt.legend()
    plt.title("Objective gap vs. iteration")
    plt.show()

    # 4. report table
    print("\nSummary:")
    print(f"{'μ':>10} | {'iters':>6} | {'sparsity(%)':>12}")
    print("-" * 32)
    for mu in mu_list:
        iters = results[mu]['hist']['iters']
        spars = results[mu]['sparsity'] * 100
        print(f"{mu:10.3g} | {iters:6d} | {spars:11.2f}")

    return results


if __name__ == "__main__":
    # ---- parameters you may want to tweak --------------------------------
    MU_LIST = (1e-3, 1e-2, 5e-2)      # PY syntax: 5e-2 == 0.05
    NEWTON_OPTS = dict(tol=1e-8, max_iter=50, alpha=0.01, beta=0.5)
    RHO = 1.0                         # ADMM penalty (can tune with residual balancing)
    DATA_PATH = "a9a.txt"            # if you placed the file locally
    # ---------------------------------------------------------------------
    run_experiment(mu_list=MU_LIST, rho=RHO,
                   newton_params=NEWTON_OPTS, data_path=DATA_PATH)
