# Derived from pytorch-minimize (MIT, (c) 2021 Reuben Feinman); see LICENSE.
import torch
from scipy.optimize import OptimizeResult

from .function import ScalarFunction
from .wolfe_line_search import strong_wolfe

_status_message = {
    "success": "Optimization terminated successfully.",
    "maxiter": "Maximum number of iterations has been exceeded.",
    "callback_stop": "Stopped by the user through the callback function.",
}


dot = lambda u, v: torch.dot(u.view(-1), v.view(-1))


@torch.no_grad()
def _minimize_cg(
    x0,
    max_iter=None,
    fun_and_grad=None,
    tol=1e-4,
    gtol=1e-5,
    callback=None,
    verbose=False,
):
    """Minimize a scalar function of one or more variables using
    nonlinear conjugate gradient.

    The algorithm is described in Nocedal & Wright (2006) chapter 5.2.

    Parameters
    ----------
    x0 : Tensor
        Initialization point.
    fun_and_grad : callable
        ``fun_and_grad(x) -> (scalar tensor, grad tensor)`` — joint value +
        gradient evaluation.
    max_iter : int
        Maximum number of iterations to perform. Default 200.
    tol : float
        Relative iterate-change stopping tolerance.
    gtol : float
        Relative gradient-norm stopping tolerance: converges when
        ``‖g_k‖ / ‖g_0‖ ≤ gtol``.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``
    verbose : bool
        If True, print status messages.
    """
    if max_iter is None:
        max_iter = 200

    # Construct scalar objective function
    sf = ScalarFunction(x0.shape, fun_and_grad)

    # initialize
    x = x0.detach().flatten().clone()
    f, g = sf.closure(x)
    if verbose:
        print("initial fval: %0.4f" % f)
    d = g.neg()
    grad_norm = g.norm()
    g_norm_0 = grad_norm.clone().clamp(min=1e-12)
    old_f = f + grad_norm.item() / 2  # Sets the initial step guess to dx ~ 1

    cached_step = [None]
    for niter in range(1, max_iter + 1):
        cached_step[0] = None
        # delta/gtd
        delta = grad_norm.pow(2)
        gtd = dot(g, d)

        # compute initial step guess based on (f - old_f) / gtd
        t0 = torch.clamp(2.02 * (f - old_f) / gtd, max=1.0)
        if t0 <= 0:
            warnflag = 4
            msg = "Initial step guess is negative."
            break
        old_f = f

        def polak_ribiere_powell_step(t, g_next):
            y = g_next - g
            beta = torch.clamp(dot(y, g_next) / delta, min=0)
            d_next = g_next.neg().add_(d, alpha=beta.item())
            torch.norm(g_next, out=grad_norm)
            return t, d_next

        def descent_condition(t, f_next, g_next):
            # Polak-Ribiere+ needs an explicit check of a sufficient
            # descent condition, which is not guaranteed by strong Wolfe.
            cached_step[:] = polak_ribiere_powell_step(t, g_next)
            t, d_next = cached_step

            # Accept step if it leads to convergence.
            cond1 = grad_norm / g_norm_0 <= gtol

            # Accept step if sufficient descent condition applies.
            cond2 = dot(d_next, g_next) <= -0.01 * grad_norm.pow(2)

            return cond1 | cond2

        # Perform CG step
        f, g, t = strong_wolfe(
            sf.dir_evaluate, x, t0, d, f, g, gtd, c2=0.4, extra_condition=descent_condition
        )

        # Update x and then update d (in that order)
        step = d.mul(t)
        x.add_(step)
        if t == cached_step[0]:
            # Reuse already computed results if possible
            d = cached_step[1]
        else:
            d = polak_ribiere_powell_step(t, g)[1]

        if verbose:
            print("iter %3d - fval: %0.4f" % (niter, f))
        if callback is not None:
            if callback(x):
                warnflag = 5
                msg = _status_message["callback_stop"]
                break

        # check relative iterate change
        if step.norm() / x.norm().clamp(min=1e-12) <= tol:
            warnflag = 0
            msg = _status_message["success"]
            break

        # convergence by 1st-order optimality (relative to initial gradient)
        if grad_norm / g_norm_0 <= gtol:
            warnflag = 0
            msg = _status_message["success"]
            break

    else:
        # if we get to the end, the maximum iterations was reached
        warnflag = 1
        msg = _status_message["maxiter"]

    if verbose:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % niter)
        print("         Function evaluations: %d" % sf.nfev)

    result = OptimizeResult(
        fun=f,
        x=x.view_as(x0),
        grad=g.view_as(x0),
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        nit=niter,
        nfev=sf.nfev,
    )
    return result
