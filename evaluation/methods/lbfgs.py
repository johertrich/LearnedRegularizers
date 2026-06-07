# Derived from pytorch-minimize (MIT, (c) 2021 Reuben Feinman); see LICENSE.
from collections import deque
import torch
from scipy.optimize import OptimizeResult

from .function import ScalarFunction
from .wolfe_line_search import strong_wolfe
from .weak_wolfe_line_search import weak_wolfe

_status_message = {
    "success": "Optimization terminated successfully.",
    "maxiter": "Maximum number of iterations has been exceeded.",
    "pr_loss": "Desired error not necessarily achieved due to precision loss.",
    "callback_stop": "Stopped by the user through the callback function.",
}


class L_BFGS:
    def __init__(self, history_size=10):
        self.history = deque(maxlen=history_size)
        self.H_diag = 1.0

    def solve(self, grad):
        d = grad.neg()
        alphas = []
        for s_i, y_i, rho_i in reversed(self.history):
            a = rho_i * s_i.dot(d)
            alphas.append(a)
            d.add_(y_i, alpha=-a.item())
        d.mul_(self.H_diag)
        for (s_i, y_i, rho_i), a in zip(self.history, reversed(alphas)):
            beta_i = rho_i * y_i.dot(d)
            d.add_(s_i, alpha=(a - beta_i).item())
        return d

    def update(self, s, y, g_old, t, damping_eps=0.2):
        """Update curvature history with pair (s, y).

        Parameters
        ----------
        s : Tensor
            Iterate change ``x_{k+1} - x_k``.
        y : Tensor
            Gradient change ``g_{k+1} - g_k``.
        g_old : Tensor
            Gradient at ``x_k`` (before the step).  Used to form the exact
            ``B s = -t g_old`` (valid because the L-BFGS direction satisfies
            ``B d = -g``, so ``B s = B(td) = -t g``).
        t : float
            Accepted step size from the line search.
        damping_eps : float or None
            Powell damping threshold (Nocedal & Wright §18.3, recommended 0.2).
            When ``y^T s < damping_eps * s^T B s``, ``y`` is blended toward
            ``Bs`` so the curvature condition is met instead of skipping the
            pair.  Set to ``None`` to disable damping and use the plain skip.
        """
        rho_inv = y.dot(s)
        if damping_eps is not None:
            Bs = g_old.mul(-t)
            sBs = s.dot(Bs)
            if rho_inv < damping_eps * sBs:
                theta = (1.0 - damping_eps) * sBs / (sBs - rho_inv)
                y = theta * y + (1.0 - theta) * Bs
                rho_inv = y.dot(s)
        if rho_inv <= 1e-10:
            return
        self.history.append((s, y, 1.0 / rho_inv))
        self.H_diag = rho_inv / y.dot(y)


@torch.no_grad()
def _minimize_lbfgs(
    x0,
    lr=1.0,
    history_size=10,
    max_iter=200,
    fun_and_grad=None,
    tol=1e-4,
    gtol=1e-5,
    gtd_tol=1e-10,
    callback=None,
    verbose=False,
    c1=1e-4,
    c2=0.9,
    tolerance_change=1e-6,
    max_ls=25,
    line_search_variant="strong",
    damping_eps=None,
):
    """Minimize a multivariate function with L-BFGS.

    Parameters
    ----------
    x0 : Tensor
        Initialization point.
    fun_and_grad : callable
        ``fun_and_grad(x) -> (scalar tensor, grad tensor)`` — joint value +
        gradient evaluation.
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    history_size : int
        Number of curvature pairs kept in the L-BFGS memory. Default 10.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to 200.
    tol : float
        Relative iterate-change stopping tolerance.
    gtol : float
        Relative gradient-norm stopping tolerance: converges when
        ``‖g_k‖ / ‖g_0‖ ≤ gtol``.
    gtd_tol : float
        Tolerance used to verify that the search direction is a descent
        direction. The directional derivative ``gtd`` should be negative for
        descent; this check ensures that ``gtd < -gtd_tol``.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    verbose : bool
        If True, print status messages.
    line_search_variant : str
        Line search algorithm: ``'strong'`` (strong Wolfe conditions,
        recommended for full-batch deterministic optimization) or
        ``'weak'`` (weak Wolfe conditions, recommended for stochastic and
        non-convex optimization). Default: ``'strong'``.
    damping_eps : float or None
        Powell damping threshold (Nocedal & Wright §18.3).  Auto-selected
        based on ``line_search_variant`` if ``None``: disabled for strong
        Wolfe (damping rarely triggers), enabled (0.2) for weak Wolfe
        (essential for weak curvature pairs).  Pass an explicit value to
        override.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    # Auto-select damping based on line search variant
    if damping_eps is None:
        damping_eps = None if line_search_variant == "strong" else 0.2

    if line_search_variant not in ("strong", "weak"):
        raise ValueError(
            f"Invalid line_search_variant: {line_search_variant}. "
            "Must be 'strong' or 'weak'."
        )
    ls_func = strong_wolfe if line_search_variant == "strong" else weak_wolfe
    sf = ScalarFunction(x0.shape, fun_and_grad)

    x = x0.flatten().clone()
    f, g = sf.closure(x)
    g_norm_0 = g.norm().clamp(min=1e-12)
    if verbose:
        print("initial fval: %0.4f" % f)

    hess = L_BFGS(history_size)
    d = g.neg()
    t = min(1.0, 1.0 / g.norm(p=1)) * lr
    n_iter = 0

    for n_iter in range(1, max_iter + 1):

        # --- Quasi-Newton direction ---
        if n_iter > 1:
            d = hess.solve(g)

        gtd = g.dot(d)  # directional derivative; must be negative for descent
        if gtd > -gtd_tol:
            warnflag = 4
            msg = "A non-descent direction was encountered."
            break

        # --- line search ---
        f_new, g_new, t = ls_func(
            sf.dir_evaluate,
            x,
            t,
            d,
            f,
            g,
            gtd,
            c1=c1,
            c2=c2,
            tolerance_change=tolerance_change,
            max_ls=max_ls,
        )
        if verbose:
            print("iter %3d - fval: %0.4f" % (n_iter, f_new))

        # --- Hessian update ---
        s = d.mul(t)
        y = g_new.sub(g)
        hess.update(s, y, g, t, damping_eps=damping_eps)

        # --- commit state ---
        # f updated in-place: keeps the scalar tensor alive so the returned result
        # always reflects the final iterate without materialising a new tensor
        f[...] = f_new
        x.add_(s)
        g = g_new
        t = lr

        if callback is not None and callback(x):
            warnflag = 5
            msg = _status_message["callback_stop"]
            break

        # convergence by relative iterate change
        if s.norm() / x.norm().clamp(min=1e-12) <= tol:
            warnflag = 0
            msg = _status_message["success"]
            break

        # convergence by 1st-order optimality (relative to initial gradient)
        if g.norm() / g_norm_0 <= gtol:
            warnflag = 0
            msg = _status_message["success"]
            break

        # precision loss; exit
        if not f.isfinite():
            warnflag = 2
            msg = _status_message["pr_loss"]
            break

    else:
        warnflag = 1
        msg = _status_message["maxiter"]

    if verbose:
        print(msg)
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % n_iter)
        print("         Function evaluations: %d" % sf.nfev)
    return OptimizeResult(
        fun=f,
        x=x.view_as(x0),
        grad=g.view_as(x0),
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        nit=n_iter,
        nfev=sf.nfev,
    )
