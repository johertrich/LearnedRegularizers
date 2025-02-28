import torch
import numpy as np

def backtracking(f,grad_f, x_k_minus_1, x_k, t_k, tau_k, mu, rho, q_k):
    for i in range(100):  # inner loop with a max of 100 iterations
        tau_k_plus_1 = (rho ** i) * tau_k
        q_k_plus_1 = (mu * tau_k_plus_1)
        t_k_plus_1 = (1 - q_k * t_k ** 2 + np.sqrt((1 - q_k * t_k ** 2) ** 2 + 4 * (q_k / q_k_plus_1) * t_k ** 2)) / 2
        beta_k_plus_1 = (t_k - 1) / t_k_plus_1 * (1 - t_k_plus_1 * tau_k_plus_1 * mu) / (1 - tau_k_plus_1 * mu)
        y_k_plus_1 = x_k + beta_k_plus_1 * (x_k - x_k_minus_1)
        x_k_plus_1 = y_k_plus_1 - tau_k_plus_1 * grad_f(y_k_plus_1)
        x_k_plus_1 = x_k_plus_1.clamp(0, 1)

        breg_div = f(x_k_plus_1) - f(y_k_plus_1) - torch.dot(grad_f(y_k_plus_1).view(-1), (x_k_plus_1 - y_k_plus_1).view(-1))
        if breg_div <= torch.norm(x_k_plus_1 - y_k_plus_1) ** 2 / (2 * tau_k_plus_1):
            break
    return x_k, x_k_plus_1, tau_k_plus_1, t_k_plus_1, q_k_plus_1


def GFISTA_backtracking_nog_algorithm(
    x_init,
    tau_init,
    rho,
    mu,
    Niter,
    f,
    grad_f,
    tol,
    verbose,
):
    """GFISTA with backtracking (version introduced in "Backtracking strategies
    for accelerated descent methods with smooth composite objectives" 
    applied to a non-composite (or assuming that you are clamping x to (0,1), 
    then with h being indicator function) strongly convex function. 
    
    Effectively just NAG with backtracking.
    """
    x_k = x_init
    x_k_minus_1 = x_init
    tau_k = tau_init
    t_k = 1
    q_k = mu * tau_k
    for k in range(Niter):
        x_k_minus_1, x_k, tau_k, t_k, q_k = backtracking(
            f,
            grad_f,
            x_k_minus_1,
            x_k,
            t_k,
            tau_k/rho, ## increase first
            mu,
            rho,
            q_k
        )
        print(k, f(x_k), tau_k)

        res_vec = torch.norm(x_k - x_k_minus_1)/torch.norm(x_k)
        res = torch.max(res_vec)
        if res < tol:
            if(verbose): print(f"Converged in iter {k}, tol {res.item():.6f}")
            break
    if verbose and res >= tol:
        print(f"max iter reached, tol {res.item():.6f}")
    return x_k


def reconstruct_nag_backtrack(
    y,
    physics,
    data_fidelity,
    regularizer,
    lmbd,
    NAG_step_size,
    NAG_max_iter,
    NAG_tol,
    detach_grads=False,
    verbose=False,
    x_init=None,
    rho = 0.9
):
    
    def energy(val):
        return data_fidelity(val, y, physics) + lmbd * regularizer.g(val)
    def energy_grad(val):
        grad = data_fidelity.grad(val, y, physics) + lmbd * regularizer.grad(val)
        if detach_grads:
            grad = grad.detach()
        return grad
    
    if x_init is not None:
        # User-defined initialization or warm start
        x = torch.clone(x_init).detach().requires_grad_(True)
    else:
        x = physics.A_dagger(y)

    rec = GFISTA_backtracking_nog_algorithm(
        x_init = x,
        tau_init = NAG_step_size,
        rho = rho,
        mu = 1,
        Niter = NAG_max_iter,
        f = energy,
        grad_f = energy_grad,
        tol = NAG_tol,
        verbose = verbose
        )
        
    return rec

