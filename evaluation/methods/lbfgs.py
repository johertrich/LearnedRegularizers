# Derived from pytorch-minimize (MIT, (c) 2021 Reuben Feinman); see LICENSE.
from abc import ABC, abstractmethod
import torch
from scipy.optimize import OptimizeResult

from .function import ScalarFunction
from .wolfe_line_search import strong_wolfe

_status_message = {
    'success':       'Optimization terminated successfully.',
    'maxiter':       'Maximum number of iterations has been exceeded.',
    'pr_loss':       'Desired error not necessarily achieved due to precision loss.',
    'callback_stop': 'Stopped by the user through the callback function.',
}


class HessianUpdateStrategy(ABC):
    def __init__(self):
        self.n_updates = 0

    @abstractmethod
    def solve(self, grad):
        pass

    @abstractmethod
    def _update(self, s, y, rho_inv):
        pass

    def update(self, s, y):
        rho_inv = y.dot(s)
        if rho_inv <= 1e-10:
            # curvature is negative; do not update
            return
        self._update(s, y, rho_inv)
        self.n_updates += 1


class L_BFGS(HessianUpdateStrategy):
    def __init__(self, x, history_size=10):
        super().__init__()
        self.y = []
        self.s = []
        self.rho = []
        self.H_diag = 1.0
        self.alpha = x.new_empty(history_size)
        self.history_size = history_size

    def solve(self, grad):
        mem_size = len(self.y)
        d = grad.neg()
        for i in reversed(range(mem_size)):
            self.alpha[i] = self.s[i].dot(d) * self.rho[i]
            d.add_(self.y[i], alpha=-self.alpha[i])
        d.mul_(self.H_diag)
        for i in range(mem_size):
            beta_i = self.y[i].dot(d) * self.rho[i]
            d.add_(self.s[i], alpha=self.alpha[i] - beta_i)

        return d

    def _update(self, s, y, rho_inv):
        if len(self.y) == self.history_size:
            self.y.pop(0)
            self.s.pop(0)
            self.rho.pop(0)
        self.y.append(y)
        self.s.append(s)
        self.rho.append(rho_inv.reciprocal())
        self.H_diag = rho_inv / y.dot(y)


@torch.no_grad()
def _minimize_lbfgs(
    x0,
    lr=1.0,
    history_size=15,
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

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    lr = float(lr)

    # construct scalar objective function
    sf = ScalarFunction(x0.shape, fun_and_grad)
    closure = sf.closure
    dir_evaluate = sf.dir_evaluate

    # compute initial f(x) and f'(x)
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)
    f, g = closure(x)
    g_norm_0 = g.norm().clamp(min=1e-12)
    if verbose:
        print("initial fval: %0.4f" % f)

    # initial settings
    hess = L_BFGS(x, history_size)
    d = g.neg()
    t = min(1.0, g.norm(p=1).reciprocal()) * lr
    n_iter = 0

    # L-BFGS iterations
    for n_iter in range(1, max_iter + 1):

        # ==================================
        #   compute Quasi-Newton direction
        # ==================================

        if n_iter > 1:
            d = hess.solve(g)

        # directional derivative
        gtd = g.dot(d)

        # check if directional derivative is below tolerance
        if gtd > -gtd_tol:
            warnflag = 4
            msg = "A non-descent direction was encountered."
            break

        # ======================
        #   update parameter
        # ======================

        f_new, g_new, t, ls_evals = strong_wolfe(
            dir_evaluate, x, t, d, f, g, gtd,
            c1=c1, c2=c2, tolerance_change=tolerance_change, max_ls=max_ls,
        )
        x_new = x + d.mul(t)

        if verbose:
            print("iter %3d - fval: %0.4f" % (n_iter, f_new))
        if callback is not None:
            if callback(x_new):
                warnflag = 5
                msg = _status_message["callback_stop"]
                break

        # ================================
        #   update hessian approximation
        # ================================

        s = x_new.sub(x)
        y = g_new.sub(g)

        hess.update(s, y)

        # =========================================
        #   check conditions and update buffers
        # =========================================

        # convergence by relative iterate change
        if s.norm() / x_new.norm().clamp(min=1e-12) <= tol:
            warnflag = 0
            msg = _status_message["success"]
            break

        # update state
        f[...] = f_new
        x.copy_(x_new)
        g.copy_(g_new)
        t = lr

        # convergence by 1st-order optimality (relative to initial gradient)
        if g.norm() / g_norm_0 <= gtol:
            warnflag = 0
            msg = _status_message["success"]
            break

        # precision loss; exit
        if ~f.isfinite():
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
