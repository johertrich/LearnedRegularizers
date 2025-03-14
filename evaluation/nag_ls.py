import torch

"""
NAG with line search
Effectively Alg 1 in https://arxiv.org/pdf/2307.14323
"""

def reconstruct_NAG_LS(
    y,
    physics,
    data_fidelity,
    regularizer,
    lmbd,
    NAG_step_size,
    NAG_max_iter,
    NAG_tol,
    verbose=False,
    x_init=None,
    progress=False,
    rho=0.9,
    delta=0.9,
    tau_threshold=1e6
):
    # run Nesterov Accelerated Gradient
    if x_init is not None:
        # User-defined initialization or warm start
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y).detach()
    
    t = torch.ones(x.shape[0], device=x.device).view(-1,1,1,1)
    x_old = torch.clone(x)
    tau = NAG_step_size * torch.ones(x.shape[0], device=x.device).view(-1,1,1,1)

    res = (NAG_tol + 1) * torch.ones(x.shape[0], device=x.device)

    def f(x_in):
        return (data_fidelity(x_in, y, physics) + lmbd * regularizer.g(x_in)).detach()
    
    def grad_f(x_in):
        return (data_fidelity.grad(x_in, y, physics) + lmbd * regularizer.grad(x_in)).detach()
    
    def backtracking(xcurr, xold, t, taubtinit, tau, rho):
        id_bt = torch.arange(0, xcurr.shape[0], device=xcurr.device)
        tau_new = torch.clone(taubtinit)
        t_new = torch.clone(t)
        xnew = torch.clone(xcurr)
        z = torch.clone(xcurr)
        breg_div = torch.ones(xcurr.shape[0], device=xcurr.device)
        for i in range(100):
            tau_new[id_bt] = rho**i * taubtinit[id_bt]
            t_new[id_bt] = 0.5 * (1 + torch.sqrt(1 + 4 * tau[id_bt] * t[id_bt]**2 / tau_new[id_bt]))
            z[id_bt] = xcurr[id_bt] + (t[id_bt] - 1) / t_new[id_bt] * (xcurr[id_bt] - xold[id_bt])
            grad_fz = grad_f(z)
            xnew[id_bt] = z[id_bt] - tau_new[id_bt] * grad_fz[id_bt]
            breg_div = (f(xnew) - f(z) - torch.sum(grad_fz * (xnew - z), dim=(1, 2, 3)))
            condition_bt = (breg_div > torch.sum((xnew - z)**2, dim=(1, 2, 3)) / (2 * tau_new).flatten())
            id_bt = condition_bt.nonzero().view(-1)

            if id_bt.numel() == 0:
                break
        return xnew, xcurr, tau_new, t_new
    
    # The index of the images that have not converged yet
    idx = torch.arange(0, x.shape[0], device=x.device)
    # Relative change in the estimate
    res = (NAG_tol + 1) * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

    if progress:
        energy_list = [f(x).item()]
        grad_norm_list = [torch.norm(grad_f(x)).item()]
        res_list = []

    for step in range(NAG_max_iter):
        tau_bt_init = torch.min(tau[idx] / delta, torch.tensor(tau_threshold, device=x.device))
        x[idx], x_old[idx], tau[idx], t[idx] = backtracking(x[idx], x_old[idx], t[idx], tau_bt_init, tau[idx], rho)
        
        if step > 0:
            res[idx] = torch.norm(x[idx] - x_old[idx], p=2, dim=(1, 2, 3)) / torch.norm(x[idx], p=2, dim=(1, 2, 3))
        
        condition = (res > NAG_tol)
        idx = condition.nonzero().view(-1)

        if progress:
            energy_list.append(f(x).item())
            grad_norm_list.append(torch.norm(grad_f(x)).item())
            res_list.append(res.item())

        if torch.max(res) < NAG_tol:
            if verbose:
                print(f"Converged in iter {step}, tol {torch.max(res).item():.6f}")
            break
        
    if verbose and res >= NAG_tol:
        print(f"max iter reached, tol {torch.max(res).item():.6f}")
    
    return (x, energy_list, grad_norm_list, res_list) if progress else x