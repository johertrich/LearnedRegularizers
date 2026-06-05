# Derived from pytorch-minimize (MIT, (c) 2021 Reuben Feinman); see LICENSE.
from .lbfgs import _minimize_lbfgs
from .cg import _minimize_cg

_tolerance_keys = {
    "l-bfgs": "gtol",
    "cg": "gtol",
}


def minimize(
    x0,
    method="l-bfgs",
    fun_and_grad=None,
    max_iter=None,
    tol=None,
    options=None,
    callback=None,
    verbose=False,
):
    """Minimize a scalar function of one or more variables.

    .. note::
        This is a general-purpose minimizer that calls one of the available
        gradient-based routines based on a supplied `method` argument.

    Parameters
    ----------
    x0 : Tensor
        Initialization point.
    fun_and_grad : callable
        ``fun_and_grad(x) -> (scalar tensor, grad tensor)`` — joint value +
        gradient evaluation.
    method : str
        The minimization routine to use. One of ``'l-bfgs'`` (default),
        ``'cg'``.
    max_iter : int, optional
        Maximum number of iterations to perform. If unspecified, this will
        be set to the default of the selected method.
    tol : float
        Tolerance for termination. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of keyword arguments to pass to the selected minimization
        routine.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    verbose : bool
        If True, print status messages.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    """
    if method not in ("l-bfgs", "cg"):
        raise ValueError(f"Unknown method '{method}': choose from 'l-bfgs', 'cg'.")
    if options is None:
        options = {}
    if fun_and_grad is not None:
        options["fun_and_grad"] = fun_and_grad
    if tol is not None:
        options.setdefault(_tolerance_keys[method], tol)
    options.setdefault("max_iter", max_iter)
    options.setdefault("callback", callback)
    options.setdefault("verbose", verbose)

    if method == "l-bfgs":
        return _minimize_lbfgs(x0, **options)
    else:
        return _minimize_cg(x0, **options)
