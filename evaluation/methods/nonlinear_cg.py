# Derived from pytorch-minimize (MIT, (c) 2021 Reuben Feinman); see LICENSE.
import torch
from .line_search import strong_wolfe


@torch.no_grad()
def nonlinear_cg(
    fun_and_grad,
    x0,
    max_iter=200,
    tol=1e-4,
    gtol=1e-5,
    verbose=False,
):
    """Polak-Ribière+ nonlinear conjugate gradient minimizer (Nocedal & Wright §5.2). See ``reconstruct`` for parameter docs."""
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
    if verbose:
        print("initial fval: %0.4f" % f)
    d = g.neg()
    grad_norm = g.norm()
    g_norm_0 = grad_norm.clamp(min=1e-12)
    old_f = f + grad_norm.item() / 2  # Sets the initial step guess to dx ~ 1

    cached_step = None
    converged = False
    for niter in range(1, max_iter + 1):
        cached_step = None
        delta = grad_norm.pow(2)
        gtd = g.dot(d)

        # compute initial step guess based on (f - old_f) / gtd
        t0 = torch.clamp(2.02 * (f - old_f) / gtd, max=1.0)
        if t0 <= 0:
            if verbose:
                print("Initial step guess is negative.")
            break
        old_f = f

        def polak_ribiere_powell_step(t, g_next):
            y = g_next - g
            beta = torch.clamp(y.dot(g_next) / delta, min=0)
            d_next = g_next.neg().add_(d, alpha=beta.item())
            grad_norm.copy_(g_next.norm())
            return t, d_next

        def descent_condition(t, _f_next, g_next):
            nonlocal cached_step
            # Polak-Ribiere+ needs an explicit check of a sufficient
            # descent condition, which is not guaranteed by strong Wolfe.
            cached_step = polak_ribiere_powell_step(t, g_next)
            t, d_next = cached_step

            # Accept step if it leads to convergence.
            cond1 = grad_norm / g_norm_0 <= gtol

            # Accept step if sufficient descent condition applies.
            cond2 = d_next.dot(g_next) <= -0.01 * grad_norm.pow(2)

            return cond1 | cond2

        f, g, t = strong_wolfe(
            _dir_eval,
            x,
            t0,
            d,
            f,
            g,
            gtd,
            c2=0.4,
            extra_condition=descent_condition,
        )

        step = d.mul(t)
        x.add_(step)
        if cached_step is not None and t == cached_step[0]:
            d = cached_step[1]
        else:
            d = polak_ribiere_powell_step(t, g)[1]

        if step.norm() / x.norm().clamp(min=1e-12) <= tol:
            converged = True
            break

        if grad_norm / g_norm_0 <= gtol:
            converged = True
            break

    else:
        if verbose:
            print("Maximum number of iterations exceeded.")

    if verbose:
        print("Current function value: %f" % f)
        print("Iterations: %d" % niter)
        print("Function evaluations: %d" % nfev)

    return x.view_as(x0), niter, converged
