# Derived from pytorch-minimize (MIT, (c) 2021 Reuben Feinman); see LICENSE.
from collections import deque
import torch
from .line_search import strong_wolfe, weak_wolfe


class _LBFGSHessian:
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
def lbfgs(
    fun_and_grad,
    x0,
    lr=1.0,
    history_size=10,
    max_iter=200,
    tol=1e-4,
    gtol=1e-5,
    gtd_tol=1e-10,
    verbose=False,
    c1=1e-4,
    c2=0.9,
    tolerance_change=1e-6,
    max_ls=25,
    line_search_variant="strong",
    damping_eps=None,
):
    """L-BFGS minimizer (Nocedal & Wright §7.4). See ``reconstruct`` for parameter docs."""
    if damping_eps is None:
        damping_eps = None if line_search_variant == "strong" else 0.2

    if line_search_variant not in ("strong", "weak"):
        raise ValueError(
            f"Invalid line_search_variant: {line_search_variant}. "
            "Must be 'strong' or 'weak'."
        )
    ls_func = strong_wolfe if line_search_variant == "strong" else weak_wolfe
    x_shape = x0.shape
    nfev = 0

    def _eval(x_flat):
        nonlocal nfev
        nfev += 1
        f, g = fun_and_grad(x_flat.reshape(x_shape))
        return f.squeeze(), g.flatten()

    def _dir_eval(x, t, d):
        return _eval(x + d.mul(t))

    x = x0.flatten().clone()
    f, g = _eval(x)
    g_norm_0 = g.norm().clamp(min=1e-12)
    if verbose:
        print("initial fval: %0.4f" % f)

    hess = _LBFGSHessian(history_size)
    d = g.neg()
    t = min(1.0, 1.0 / g.norm(p=1)) * lr
    n_iter = 0
    converged = False

    for n_iter in range(1, max_iter + 1):

        # --- Quasi-Newton direction ---
        if n_iter > 1:
            d = hess.solve(g)

        gtd = g.dot(d)  # directional derivative; must be negative for descent
        if gtd > -gtd_tol:
            if verbose:
                print("A non-descent direction was encountered.")
            break

        # --- line search ---
        f_new, g_new, t = ls_func(
            _dir_eval,
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

        f = f_new
        x.add_(s)
        g = g_new
        t = lr

        if s.norm() / x.norm().clamp(min=1e-12) <= tol:
            converged = True
            break

        if g.norm() / g_norm_0 <= gtol:
            converged = True
            break

        if not f.isfinite():
            if verbose:
                print("Precision loss; desired accuracy not achieved.")
            break

    else:
        if verbose:
            print("Maximum number of iterations exceeded.")

    if verbose:
        print("Current function value: %f" % f)
        print("Iterations: %d" % n_iter)
        print("Function evaluations: %d" % nfev)
    return x.view_as(x0), n_iter, converged
