"""Natively batched L-BFGS for image reconstruction.

This solves ``min_x f(x, y)`` for a whole batch at once, keeping *independent*
per-sample state -- in contrast to flattening the batch into one vector and
minimising the summed energy (which couples the samples through a single step
size and a single curvature history).

Why L-BFGS batches cleanly: it is matrix-free. The search direction comes from
the two-loop recursion, which is only inner products and vector updates. With a
batch axis, every inner product becomes a per-sample reduction over the
non-batch dimensions (keeping the batch dim), so the curvature history is
``(history_size, B, C, H, W)`` and every ``rho``/``alpha``/``beta``/``gamma`` is
``(B, 1, 1, 1)``. No Hessian matrix is ever formed, and there is no per-sample
Python loop.

The genuinely sequential part is the line search. We use a *masked backtracking
Armijo* line search: a per-sample step ``t`` is shrunk only for the samples that
fail the Armijo condition, the rest are frozen, and we iterate until all active
samples pass (or a cap is hit). Converged samples are frozen (step 0) so they
neither move nor corrupt the shared control flow; this trades a little redundant
compute for fully vectorised execution.

Objective interface (identical to :func:`torchmin.nmAPG`):

* ``f(x, y) -> (B,)``
* ``nabla(x, y) -> (B, C, H, W)``
* ``f_and_nabla(x, y) -> ((B,), (B, C, H, W))``

Returns ``(x, inv_gamma, steps, converged, path)`` where ``inv_gamma`` is the
per-sample inverse of the L-BFGS initial-Hessian scaling -- a loose analogue of
nmAPG's Lipschitz estimate ``L`` (it is *not* a true Lipschitz constant), kept
in the return tuple so the ``reconstruct`` wrapper can report uniform stats.
"""

from collections import deque
import torch


@torch.no_grad()
def lbfgs_batched(
    x0,
    y,
    f,
    f_and_nabla,
    nabla=None,  # accepted for interface symmetry with nmAPG; unused
    max_iter: int = 200,
    step_size: float = 1.0,  # only scales the first (steepest-descent) step
    tol: float = 1e-4,  # iterate-change stopping tolerance (xtol)
    gtol: float = 1e-5,  # gradient-norm stopping tolerance
    history_size: int = 10,
    c1: float = 1e-4,  # Armijo sufficient-decrease constant
    backtrack: float = 0.5,  # step shrink factor
    max_ls: int = 25,  # max backtracking evaluations per iteration
    verbose: bool = False,
    return_path: bool = False,
):
    """Batched L-BFGS with masked Armijo backtracking line search.

    See module docstring for the objective interface and semantics.

    Parameters
    ----------
    tol : float
        Relative iterate-change stopping tolerance (xtol).
    gtol : float
        Relative gradient-norm stopping tolerance; a sample is considered
        converged when ``‖g_k‖ / ‖g_0‖ ≤ gtol``.  Checked after the
        iterate-change criterion so it adds no extra function evaluations.
    """
    x = x0.clone()
    B = x.shape[0]
    reduce_dims = tuple(range(1, x.ndim))

    def red(t):
        # per-sample reduction over non-batch dims, keeping broadcast shape
        return t.sum(reduce_dims, keepdim=True)

    def kd(vec_b):
        # (B,) -> (B, 1, 1, ...) so it broadcasts against x
        return vec_b.reshape(B, *([1] * (x.ndim - 1)))

    f_val, g = f_and_nabla(x, y)  # (B,), (B,C,H,W)
    f_val = f_val.reshape(B)
    g_norm_0 = g.flatten(1).norm(dim=1).clamp(min=1e-12)  # (B,)

    # L-BFGS history (oldest first). Each S[k], Y[k] is (B,C,H,W); RHO[k] is (B,1,1,1)
    S = deque(maxlen=history_size)
    Y = deque(maxlen=history_size)
    RHO = deque(maxlen=history_size)
    gamma = torch.ones(B, *([1] * (x.ndim - 1)), device=x.device, dtype=x.dtype)

    converged = torch.zeros(B, dtype=torch.bool, device=x.device)
    res = (tol + 1) * torch.ones(B, device=x.device, dtype=x.dtype)
    path = []
    it = 0

    for it in range(max_iter):
        if return_path:
            path.append(x.clone())

        # ---- two-loop recursion: d = -H_k * g  (per sample) ----
        q = g.clone()
        alphas = []
        for s_i, y_i, rho_i in zip(reversed(S), reversed(Y), reversed(RHO)):
            a = rho_i * red(s_i * q)
            q = q - a * y_i
            alphas.append(a)
        r = gamma * q
        for s_i, y_i, rho_i, a in zip(S, Y, RHO, reversed(alphas)):
            b = rho_i * red(y_i * r)
            r = r + (a - b) * s_i
        d = -r

        # safety: guarantee a descent direction per sample
        gtd = red(g * d)  # (B,1,1,1), should be < 0
        ascent = (gtd >= 0).reshape(B)
        if ascent.any():
            d = torch.where(kd(ascent), -g, d)
            gtd = red(g * d)

        # ---- per-sample initial step ----
        if len(S) == 0:
            g_l1 = red(g.abs())  # (B,1,1,1)
            t = (step_size / g_l1.clamp_min(1e-12)).clamp(max=1.0)
        else:
            t = torch.ones_like(gamma)
        t = torch.where(kd(converged), torch.zeros_like(t), t)

        # ---- masked Armijo backtracking line search ----
        gtd_b = gtd.reshape(B)
        x_new = x + t * d
        f_new = f(x_new, y).reshape(B)
        for _ls in range(max_ls):
            rhs = f_val + c1 * t.reshape(B) * gtd_b
            ok = (f_new <= rhs) | converged
            if bool(ok.all()):
                break
            shrink = ~ok
            t = torch.where(kd(shrink), t * backtrack, t)
            x_new = x + t * d
            f_new = f(x_new, y).reshape(B)

        # gradient at the accepted point
        f_new, g_new = f_and_nabla(x_new, y)
        f_new = f_new.reshape(B)

        # ---- curvature pair (per sample) ----
        s_k = x_new - x
        y_k = g_new - g
        ys = red(y_k * s_k)  # (B,1,1,1)
        yy = red(y_k * y_k)
        pos = ys > 1e-10  # accept pair only if curvature positive

        # initial-Hessian scaling for the next iteration
        gamma = torch.where(pos, ys / yy.clamp_min(1e-12), gamma)

        # store the pair; disable it per-sample (rho=0) where curvature is non-positive
        rho_k = torch.where(pos, 1.0 / ys.clamp_min(1e-12), torch.zeros_like(ys))
        S.append(s_k)
        Y.append(y_k)
        RHO.append(rho_k)
        # ---- convergence checks ----
        # (1) non-finite energy: freeze affected samples
        nonfinite = ~f_new.isfinite()
        if bool(nonfinite.any()):
            converged = converged | nonfinite
            if verbose:
                print(
                    f"iter {it}: non-finite energy in {int(nonfinite.sum())} sample(s)"
                )

        # (2) relative iterate change
        num = s_k.flatten(1).norm(dim=1)
        den = x_new.flatten(1).norm(dim=1).clamp_min(1e-12)
        step_res = num / den
        res = torch.where(converged, res, step_res)
        converged = converged | (step_res < tol)

        # (3) gradient-norm criterion (relative to initial gradient)
        g_norm = g_new.flatten(1).norm(dim=1)
        converged = converged | (g_norm / g_norm_0 <= gtol)

        # commit
        x = x_new
        g = g_new
        f_val = f_new

        if bool(converged.all()):
            if verbose:
                print(f"Converged in iter {it}, max res {float(res.max()):.6f}")
            break

    if verbose and not bool(converged.all()):
        print(f"max iter reached, max res {float(res.max()):.6f}")

    inv_gamma = (1.0 / gamma.clamp_min(1e-12)).reshape(B, 1, 1, 1)
    return x, inv_gamma, it, converged, path
