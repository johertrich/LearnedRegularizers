from collections import deque
import torch


@torch.no_grad()
def lbfgs_batched(
    x0,
    y,
    f,
    f_and_nabla,
    max_iter: int = 200,
    tol: float = 1e-4,  # iterate-change stopping tolerance (xtol)
    gtol: float = 1e-5,  # gradient-norm stopping tolerance
    history_size: int = 10,
    c1: float = 1e-4,  # Armijo sufficient-decrease constant
    backtrack: float = 0.5,  # step shrink factor
    max_ls: int = 25,  # max backtracking evaluations per iteration
    damping_eps: float = 0.2,  # Powell damping threshold; None disables
    verbose: bool = False,
):
    """Batched L-BFGS with masked Armijo backtracking line search.

    Mirrors the structure and naming of the unbatched :func:`lbfgs`
    (Nocedal & Wright §7.4) so the two can be read side by side.  The
    remaining differences are intrinsic to batching: per-sample dot products
    and norms (``red``), per-sample convergence masking (``converged``), and an
    inline Armijo backtracking line search in place of the external Wolfe one.
    Two names also differ because the symbols are taken by arguments: the energy
    *value* is ``f_val`` (``f`` is the energy *function*) and the curvature
    vector is ``y_k`` (``y`` is the measurement).

    Parameters
    ----------
    tol : float
        Relative iterate-change stopping tolerance (xtol).
    gtol : float
        Relative gradient-norm stopping tolerance; a sample is considered
        converged when ``‖g_k‖ / ‖g_0‖ ≤ gtol``.  Checked after the
        iterate-change criterion so it adds no extra function evaluations.
    damping_eps : float or None
        Powell-damping threshold (Nocedal & Wright §18.3).  The Armijo
        backtracking line search does not enforce the curvature condition.
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

    f_val, g = f_and_nabla(x, y)
    f_val = f_val.reshape(B)
    g_norm_0 = g.flatten(1).norm(dim=1).clamp(min=1e-12)
    if verbose:
        print("initial fval: %0.4f" % float(f_val.mean()))

    # Inverse-Hessian state for the L-BFGS two-loop recursion.
    history = deque(maxlen=history_size)
    H_diag = torch.ones(B, *([1] * (x.ndim - 1)), device=x.device, dtype=x.dtype)
    d = g.neg()
    t = (1.0 / red(g.abs()).clamp_min(1e-12)).clamp(max=1.0)
    n_iter = 0
    converged = torch.zeros(B, dtype=torch.bool, device=x.device)
    res = (tol + 1) * torch.ones(B, device=x.device, dtype=x.dtype)

    for n_iter in range(1, max_iter + 1):
        # --- Quasi-Newton direction (two-loop recursion: d = -H g) ---
        # Same recursion and in-place updates as the unbatched solver, but with
        # per-sample dot products via red() (a/beta_i broadcast over the batch).
        if n_iter > 1:
            d = g.neg()
            alphas = []
            for s_i, y_i, rho_i in reversed(history):
                a = rho_i * red(s_i * d)
                alphas.append(a)
                d.addcmul_(y_i, a, value=-1.0)  # d -= a * y_i (in-place, no temp)
            d.mul_(H_diag)
            for (s_i, y_i, rho_i), a in zip(history, reversed(alphas)):
                beta_i = rho_i * red(y_i * d)
                d.addcmul_(s_i, a - beta_i)  # d += (a - beta_i) * s_i

        gtd = red(g * d)  # directional derivative; must be negative for descent
        # The unbatched solver breaks on a non-descent direction; here we fall
        # back to steepest descent for any offending sample instead.
        # Unconditional so it stays a single GPU op (no host sync): a no-op
        # where nothing ascends.
        ascent = (gtd >= 0).reshape(B)
        d = torch.where(kd(ascent), g.neg(), d)
        gtd = red(g * d)

        # --- line search (masked Armijo backtracking) ---
        t = torch.where(kd(converged), torch.zeros_like(t), t)  # freeze converged
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

        # value and gradient at the accepted point
        f_new, g_new = f_and_nabla(x_new, y)
        f_new = f_new.reshape(B)

        # --- Hessian update: store curvature pair (s, y) ---
        s = x_new - x  # == d.mul(t)
        y_k = g_new - g
        rho_inv = red(y_k * s)
        if damping_eps is not None:
            # Powell damping (Nocedal & Wright §18.3): where y^T s is too small,
            # blend y toward Bs = -t g (exact since the L-BFGS direction gives
            # B d = -g, hence B s = -t g) so y^T s > 0. Per sample via masking.
            Bs = g.mul(-t)
            sBs = red(s * Bs)
            damp = rho_inv < damping_eps * sBs
            theta = (1.0 - damping_eps) * sBs / (sBs - rho_inv).clamp_min(1e-12)
            y_k = torch.where(damp, theta * y_k + (1.0 - theta) * Bs, y_k)
            rho_inv = red(y_k * s)
        # Frozen samples append with rho=0 (inert in the two-loop) and
        # leave H_diag unchanged for those samples.
        pos = rho_inv > 1e-10
        H_diag = torch.where(pos, rho_inv / red(y_k * y_k).clamp_min(1e-12), H_diag)
        rho = torch.where(
            pos, 1.0 / rho_inv.clamp_min(1e-12), torch.zeros_like(rho_inv)
        )
        history.append((s, y_k, rho))

        f_val = f_new
        x = x_new
        g = g_new
        t = torch.ones_like(H_diag)

        # --- convergence checks (per sample; cumulative, monotone) ---
        step_res = s.flatten(1).norm(dim=1) / x.flatten(1).norm(dim=1).clamp_min(1e-12)
        if verbose:  # res is only ever read in the verbose prints below
            res = torch.where(converged, res, step_res)
        converged = converged | (step_res <= tol)

        converged = converged | (g.flatten(1).norm(dim=1) / g_norm_0 <= gtol)

        # Freeze any sample whose energy went non-finite.
        converged = converged | ~f_val.isfinite()

        if bool(converged.all()):
            if verbose:
                print(f"Converged at iter {n_iter}, max res {float(res.max()):.6f}")
            break

    else:
        if verbose:
            print(
                f"Maximum number of iterations exceeded (max res {float(res.max()):.6f})."
            )

    inv_H_diag = 1.0 / H_diag.clamp_min(1e-12)  # (B, 1, 1, ...), broadcasts against x
    return x, inv_H_diag, n_iter, converged
