import inspect

import torch

from .methods import nmAPG, lbfgs_batched, adam
from .methods.lbfgs import lbfgs as lbfgs_solver
from .methods.nonlinear_cg import nonlinear_cg as cg_solver

_PER_SAMPLE = ("l-bfgs", "cg")


def reconstruct(
    y,
    physics,
    data_fidelity,
    regularizer,
    lmbd,
    step_size,
    max_iter,
    tol,
    method="nmapg",
    x_init=None,
    detach_grads=True,
    verbose=False,
    return_stats=False,
    **kwargs,
):
    """Minimise ``data_fidelity(x, y; physics) + lmbd * regularizer(x)``.

    Dispatches to the chosen optimiser:

        'nmapg', 'lbfgs_batched'  -- natively batched (one call per batch)
        'adam'                    -- Adam + cosine annealing (autograd-based)
        'l-bfgs', 'cg'            -- per-sample loop with analytic gradients

    The regularizer must expose ``g(x)`` (value) and ``grad(x)`` (gradient).
    If ``grad`` also accepts ``get_energy=True``, value and gradient are
    obtained in a single call.

    Parameters
    ----------
    y : Tensor
        Measurements.
    physics : deepinv physics object
        Defines the forward operator and noise model.
    data_fidelity : deepinv data fidelity object
        Data fidelity term of the variational problem.
    regularizer : object
        Must expose ``g(x)`` (value) and ``grad(x)`` (gradient). If
        ``grad`` also accepts ``get_energy=True`` it is used to obtain value
        and gradient in a single call.
    lmbd : float
        Regularisation weight.
    step_size : float
        Initial step size.  Interpretation is method-specific: for *nmapg*
        ``L_init`` defaults to ``1 / step_size``; for *adam* it is the
        initial Adam learning rate.  Not used by *lbfgs_batched*.
    max_iter : int
        Maximum number of solver iterations.
    tol : float
        Relative iterate-change stopping tolerance.
    method : str
        Solver to use.  One of ``'nmapg'``, ``'adam'``, ``'lbfgs_batched'``,
        ``'l-bfgs'``, ``'cg'``.
    x_init : Tensor, optional
        Warm-start iterate.  Defaults to ``physics.A_dagger(y)``.
    detach_grads : bool
        Whether to detach gradients from the computational graph (ignored for
        *adam* which needs the graph).
    verbose : bool
        Print solver progress.
    return_stats : bool
        If True, return ``(rec, stats)`` instead of just ``rec``.
    **kwargs
        Method-specific hyperparameters.  Unknown keys are forwarded as-is
        and will raise ``TypeError`` if the underlying solver does not accept
        them.

        nmapg
            ``L_init`` : float, default ``1 / step_size``
                Initial Lipschitz estimate used in the gradient step.
                Overrides the ``step_size``-derived default when provided.
            ``rho`` : float, default 0.9
                Line-search contraction factor.
            ``delta`` : float, default 0.1
                Sufficient-decrease margin in the line search.
            ``eta`` : float, default 0.8
                Momentum extrapolation damping.

        lbfgs_batched
            ``history_size`` : int, default 10
                Number of curvature pairs kept in the L-BFGS memory.
            ``c1`` : float, default 1e-4
                Armijo sufficient-decrease constant.
            ``backtrack`` : float, default 0.5
                Step shrink factor per backtracking trial.
            ``max_ls`` : int, default 25
                Maximum backtracking evaluations per iteration.
            ``gtol`` : float, default 1e-5
                Relative gradient-norm tolerance: ``‖g_k‖ / ‖g_0‖ ≤ gtol``.
            ``damping_eps`` : float or None, default 0.2
                Powell-damping threshold; ``None`` disables damping.

        adam
            No additional kwargs beyond the shared ``step_size``.

        l-bfgs / cg
            ``history_size`` : int (l-bfgs only), default 10
                L-BFGS memory size.
            ``line_search_variant`` : str (l-bfgs only), default ``'strong'``
                Line search algorithm: ``'strong'`` (strong Wolfe conditions)
                or ``'weak'`` (weak Wolfe conditions).
            ``c1`` : float (l-bfgs only), default 1e-4
                Armijo sufficient-decrease constant for the Wolfe line search.
            ``c2`` : float (l-bfgs only), default 0.9
                Curvature condition constant for the Wolfe line search.
            ``tolerance_change`` : float (l-bfgs only), default 1e-6
                Line-search bracket width below which the search terminates.
            ``max_ls`` : int (l-bfgs only), default 25
                Maximum number of Wolfe line search iterations.
            ``gtol`` : float, default 1e-5
                Relative gradient-norm tolerance: ``‖g_k‖ / ‖g_0‖ ≤ gtol``.
            ``gtd_tol`` : float (l-bfgs only), default 1e-10
                Minimum directional derivative; guards against near-zero
                descent directions.
            ``damping_eps`` : float or None (l-bfgs only), default auto
                Powell damping threshold (Nocedal & Wright §18.3). Auto-selected
                based on ``line_search_variant``: disabled for ``'strong'`` Wolfe,
                set to 0.2 for ``'weak'`` Wolfe. Pass an explicit value to override.
    """
    x = torch.clone(x_init).detach() if x_init is not None else physics.A_dagger(y)

    def energy(val, y_in):
        with torch.no_grad():
            fun = data_fidelity(val, y_in, physics) + lmbd * regularizer.g(val)
        return (fun.detach() if detach_grads else fun).reshape(-1)

    def energy_grad(val, y_in):
        grad = data_fidelity.grad(val, y_in, physics) + lmbd * regularizer.grad(val)
        return grad.detach() if detach_grads else grad

    # check if value and gradient can be obtained in a single regularizer call
    has_get_energy = "get_energy" in inspect.signature(regularizer.grad).parameters

    def energy_and_grad(val, y_in):
        if has_get_energy:
            reg_f, grad = regularizer.grad(val, get_energy=True)
            fun = data_fidelity(val, y_in, physics) + lmbd * reg_f
            grad = data_fidelity.grad(val, y_in, physics) + lmbd * grad
            if detach_grads:
                fun, grad = fun.detach(), grad.detach()
            return fun.reshape(-1), grad
        return energy(val, y_in), energy_grad(val, y_in)

    L_est = None

    if method == "nmapg":
        L_init = kwargs.pop("L_init", 1 / step_size)
        rec, L_est, steps, converged = nmAPG(
            x0=x,
            y=y,
            f=energy,
            nabla=energy_grad,
            f_and_nabla=energy_and_grad,
            max_iter=max_iter,
            L_init=L_init,
            tol=tol,
            verbose=verbose,
            **kwargs,
        )

    elif method == "adam":
        # Adam differentiates through the energy, so it needs a graph-building
        # objective (no torch.no_grad / detach here).
        def energy_diff(val, y_in):
            return data_fidelity(val, y_in, physics) + lmbd * regularizer.g(val)

        rec, steps, converged = adam(
            x0=x,
            y=y,
            energy=energy_diff,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            **kwargs,
        )

    elif method == "lbfgs_batched":
        rec, L_est, steps, converged = lbfgs_batched(
            x0=x,
            y=y,
            f=energy,
            f_and_nabla=energy_and_grad,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            **kwargs,
        )

    elif method in _PER_SAMPLE:
        # no natural batching: optimise each sample independently
        recs, steps, conv = [], [], []
        _solver = lbfgs_solver if method == "l-bfgs" else cg_solver
        for b in range(y.shape[0]):
            y_b = y[b : b + 1]
            fag = lambda x, y_b=y_b: energy_and_grad(x, y_b)
            x_b, nit, success = _solver(
                fag,
                x[b : b + 1],
                tol=tol,
                max_iter=max_iter,
                verbose=verbose,
                **kwargs,
            )
            recs.append(x_b)
            steps.append(int(nit))
            conv.append(bool(success))
        rec = torch.cat(recs)
        converged = torch.tensor(conv, device=y.device)

    if not return_stats:
        return rec
    stats = {
        "L": L_est.detach() if torch.is_tensor(L_est) else L_est,
        "steps": steps,
        "converged": converged,
    }
    return rec, stats
