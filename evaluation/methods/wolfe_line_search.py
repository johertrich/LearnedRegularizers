# Derived from pytorch-minimize (MIT, (c) 2021 Reuben Feinman); see LICENSE.
# Strong-Wolfe line search with optional extra acceptance condition.
#
# _cubic_interpolate is ported from:
#   https://github.com/torch/optim/blob/master/polyinterp.lua
#   Copyright (c) 2016 Facebook, Inc. (BSD license)
import torch

__all__ = ["strong_wolfe"]


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def strong_wolfe(
    obj_func,
    x,
    t,
    d,
    f,
    g,
    gtd=None,
    c1=1e-4,
    c2=0.9,
    tolerance_change=1e-6,
    max_ls=25,
    extra_condition=None,
):
    """Strong-Wolfe line search.

    ``obj_func(x, t, d)`` returns ``(f_new, g_new)`` at ``x + t * d``.
    ``extra_condition(t, f_new, g_new) -> bool`` is an optional additional
    acceptance check required by methods such as Polak-Ribière CG where the
    standard Wolfe conditions alone do not guarantee a descent direction.
    """
    if gtd is None:
        gtd = g.mul(d).sum()
    f, t = float(f), float(t)
    # clone g to avoid aliasing the caller's tensor; x and d are read-only
    g = g.clone(memory_format=torch.contiguous_format)

    if extra_condition is None:
        extra_condition = lambda *args: True

    d_norm = d.abs().max()
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0

    while ls_iter < max_ls:
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd and extra_condition(t, f_new, g_new):
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new  # g_new is a fresh tensor each call; no clone needed
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]
        bracket_gtd = [gtd, gtd_new]  # initialise so zoom phase is always safe

    insuf_progress = False
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)

    while not done and ls_iter < max_ls:
        b_min, b_max = min(bracket), max(bracket)
        if (b_max - b_min) * d_norm < tolerance_change:
            break

        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        eps = 0.1 * (b_max - b_min)
        if min(b_max - t, t - b_min) < eps:
            if insuf_progress or t >= b_max or t <= b_min:
                if abs(t - b_max) < abs(t - b_min):
                    t = b_max - eps
                else:
                    t = b_min + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd and extra_condition(t, f_new, g_new):
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new
            bracket_gtd[low_pos] = gtd_new

    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]

    return torch.as_tensor(f_new, dtype=x.dtype, device=x.device), g_new, t
