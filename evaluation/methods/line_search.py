# Derived from pytorch-minimize (MIT, (c) 2021 Reuben Feinman); see LICENSE.
# _cubic_interpolate is ported from:
#   https://github.com/torch/optim/blob/master/polyinterp.lua
#   Copyright (c) 2016 Facebook, Inc. (BSD license)
from collections import namedtuple
import torch

__all__ = ["strong_wolfe", "weak_wolfe"]

_BP = namedtuple("_BP", ["t", "f", "g", "gtd"])


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


def _zoom(
    obj_func,
    x,
    d,
    f,
    gtd,
    c1,
    d_norm,
    tolerance_change,
    lo,
    hi,
    extra_condition,
    curvature_cond,
    budget,
):
    """Bracket-narrowing zoom phase shared by strong and weak Wolfe line searches.

    ``curvature_cond(gtd_new) -> bool`` encodes the variant-specific curvature
    check; everything else is identical between the two variants.
    """
    insuf_progress = False
    for _ in range(budget):
        b_min, b_max = min(lo.t, hi.t), max(lo.t, hi.t)
        if (b_max - b_min) * d_norm < tolerance_change:
            break

        t = _cubic_interpolate(lo.t, lo.f, lo.gtd, hi.t, hi.f, hi.gtd)

        eps = 0.1 * (b_max - b_min)
        if min(b_max - t, t - b_min) < eps:
            if insuf_progress or t >= b_max or t <= b_min:
                t = (b_max - eps) if abs(t - b_max) < abs(t - b_min) else (b_min + eps)
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        f_new, g_new = obj_func(x, t, d)
        gtd_new = g_new.dot(d)

        if f_new > (f + c1 * t * gtd) or f_new >= lo.f:
            hi = _BP(t, f_new, g_new, gtd_new)
            if hi.f < lo.f:
                lo, hi = hi, lo
        else:
            accept = curvature_cond(gtd_new) and (
                extra_condition is None or extra_condition(t, f_new, g_new)
            )
            if not accept and gtd_new * (hi.t - lo.t) >= 0:
                hi = lo  # uses old lo, before the update below
            lo = _BP(t, f_new, g_new, gtd_new)
            if accept:
                break
    return lo


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
    f = float(f)
    g = g.clone(memory_format=torch.contiguous_format)

    d_norm = d.abs().max()
    f_new, g_new = obj_func(x, t, d)
    gtd_new = g_new.dot(d)

    prev = _BP(0, f, g, gtd)
    curr = _BP(t, f_new, g_new, gtd_new)
    done = False
    ls_iter = 0

    while ls_iter < max_ls:
        if curr.f > (f + c1 * curr.t * gtd) or (ls_iter > 1 and curr.f >= prev.f):
            lo, hi = (prev, curr) if prev.f <= curr.f else (curr, prev)
            break

        if abs(curr.gtd) <= -c2 * gtd and (
            extra_condition is None or extra_condition(curr.t, curr.f, curr.g)
        ):
            lo = curr
            done = True
            break

        if curr.gtd >= 0:
            lo, hi = (prev, curr) if prev.f <= curr.f else (curr, prev)
            break

        min_step = curr.t + 0.01 * (curr.t - prev.t)
        max_step = curr.t * 10
        t = _cubic_interpolate(
            prev.t,
            prev.f,
            prev.gtd,
            curr.t,
            curr.f,
            curr.gtd,
            bounds=(min_step, max_step),
        )
        prev = curr
        f_new, g_new = obj_func(x, t, d)
        gtd_new = g_new.dot(d)
        curr = _BP(t, f_new, g_new, gtd_new)
        ls_iter += 1

    if ls_iter == max_ls:
        st = _BP(0, f, g, gtd)
        lo, hi = (st, curr) if st.f <= curr.f else (curr, st)

    if not done:
        lo = _zoom(
            obj_func,
            x,
            d,
            f,
            gtd,
            c1,
            d_norm,
            tolerance_change,
            lo,
            hi,
            extra_condition,
            curvature_cond=lambda gtd_new: abs(gtd_new) <= -c2 * gtd,
            budget=max_ls - ls_iter,
        )
    return torch.as_tensor(lo.f, dtype=x.dtype, device=x.device), lo.g, lo.t


def weak_wolfe(
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
    """Weak-Wolfe line search.

    ``obj_func(x, t, d)`` returns ``(f_new, g_new)`` at ``x + t * d``.
    ``extra_condition(t, f_new, g_new) -> bool`` is an optional additional
    acceptance check.

    Weak Wolfe differs from strong Wolfe in the curvature condition:
      - Strong: ``|g_new^T d| <= -c2 * g^T d``
      - Weak:   ``g_new^T d >= c2 * g^T d``

    Weak Wolfe is preferred for non-convex optimization because
    it permits weaker curvature, which is paired well with Powell
    damping to ensure positive-definite Hessian updates.
    """
    if gtd is None:
        gtd = g.mul(d).sum()
    f = float(f)
    g = g.clone(memory_format=torch.contiguous_format)

    d_norm = d.abs().max()
    f_new, g_new = obj_func(x, t, d)
    gtd_new = g_new.dot(d)

    prev = _BP(0, f, g, gtd)
    curr = _BP(t, f_new, g_new, gtd_new)
    done = False
    ls_iter = 0

    while ls_iter < max_ls:
        if curr.f > (f + c1 * curr.t * gtd) or (ls_iter > 1 and curr.f >= prev.f):
            lo, hi = (prev, curr) if prev.f <= curr.f else (curr, prev)
            break

        # Weak curvature: g_new^T d >= c2 * g^T d (not absolute value)
        if curr.gtd >= c2 * gtd and (
            extra_condition is None or extra_condition(curr.t, curr.f, curr.g)
        ):
            lo = curr
            done = True
            break

        min_step = curr.t + 0.01 * (curr.t - prev.t)
        max_step = curr.t * 10
        t = _cubic_interpolate(
            prev.t,
            prev.f,
            prev.gtd,
            curr.t,
            curr.f,
            curr.gtd,
            bounds=(min_step, max_step),
        )
        prev = curr
        f_new, g_new = obj_func(x, t, d)
        gtd_new = g_new.dot(d)
        curr = _BP(t, f_new, g_new, gtd_new)
        ls_iter += 1

    if ls_iter == max_ls:
        st = _BP(0, f, g, gtd)
        lo, hi = (st, curr) if st.f <= curr.f else (curr, st)

    if not done:
        lo = _zoom(
            obj_func,
            x,
            d,
            f,
            gtd,
            c1,
            d_norm,
            tolerance_change,
            lo,
            hi,
            extra_condition,
            curvature_cond=lambda gtd_new: gtd_new >= c2 * gtd,
            budget=max_ls - ls_iter,
        )
    return torch.as_tensor(lo.f, dtype=x.dtype, device=x.device), lo.g, lo.t
