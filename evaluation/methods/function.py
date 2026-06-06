__all__ = ["ScalarFunction"]


class ScalarFunction:
    """Wraps an analytic fun_and_grad callable for gradient-based optimization.

    Parameters
    ----------
    x_shape : tuple
        Shape of the (unflattened) input ``x``.
    fun_and_grad : callable
        ``fun_and_grad(x) -> (scalar tensor, grad tensor)`` — joint value +
        gradient evaluation.  ``x`` arrives shaped as ``x_shape``; both
        outputs must be detached.
    """

    def __init__(self, x_shape, fun_and_grad):
        self._fun_and_grad = fun_and_grad
        self._x_shape = x_shape
        self.nfev = 0

    def closure(self, x):
        """Return (f, grad) at x.  Core call used by quasi-Newton / CG loops."""
        self.nfev += 1
        f, grad = self._fun_and_grad(x.detach().reshape(self._x_shape))
        return f.reshape(-1).sum(), grad.reshape(-1)

    def dir_evaluate(self, x, t, d):
        """Return (f, grad) at x + t*d.  Used by the strong-Wolfe line search."""
        self.nfev += 1
        f, grad = self._fun_and_grad((x + d.mul(t)).detach().reshape(self._x_shape))
        return f.reshape(-1).sum(), grad.reshape(-1)
